import time

import torch
import torch.nn as nn
import tqdm
from slim_gptq import SliMGPTQ
from utils.mixed_quantizer import Quantizer
from modelutils import find_layers


def get_model(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    name = model
    if "opt" in model:
        from transformers import OPTForCausalLM

        model = OPTForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = model.config.max_position_embeddings
    elif "llama" in model:
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
        if "llama-3" in name.lower():
            model.seqlen = 2048
        else:
            model.seqlen = 2048
    return model


'''
The function is employed to calibrate and quantize models layer by layer.
'''
@torch.no_grad()
def quant_sequential(model, dataloader, dev, saved_block_precision):
    print("Starting ...")

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            dev
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif "llama" in args.model:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    if "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif "llama" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")
    index = 0
    mixed_block_precision = {}
    quantizers = {}
    for i in tqdm.tqdm(range(len(layers)), desc="Running Mixed-Precision Quantization"):
        layer = layers[i].to(dev)

        subset = find_layers(layer)

        gptq = {}
        for name in subset:
            index += 1
            if (
                not (args.minlayer <= i < args.maxlayer and args.quant_only in name)
            ) == (not args.invert):
                continue
            quantizer = Quantizer(
                subset[name].weight,
                method=args.low_quant_method,
                groupsize=groupsize,
                metric = args.quantizer_matric,
                lambda_salience=args.lambda_salience        
            )
            gptq[name] = SliMGPTQ(
                subset[name],
                quantizer,
                disable_gptq=args.disable_gptq,
                layer_index=index,
                salient_block = args.salient_block,
                nonsalient_block = args.nonsalient_block,
                bit_width = int(args.low_quant_method[0])
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gptq:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        # for Activation Aware mixed-precision blocks determination
        if saved_block_precision is None and args.salient_block == -1:
            for name in gptq:
                gptq[name].get_salience(blocksize=args.groupsize)

            def get_block(name):
                def tmp(_, inp, out):
                    gptq[name].get_block(inp[0].data, out.data, blocksize=args.groupsize,)
                return tmp
            handles = []
            for name in gptq:
                handles.append(subset[name].register_forward_hook(get_block(name)))
            for j in tqdm.tqdm(range(args.nsamples), desc="Determining block precision..."):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()
        # end

        mixed_block_precision[i] = {}

        for name in gptq:
            print(i, name)
            print("Quantizing ...")
            layer_block_precision, scales, zeros, g_idx = gptq[name].fasterquant(
                percdamp=args.percdamp, 
                blocksize=args.groupsize,
                layer_name=name,
                saved_block_precision=saved_block_precision[i][name] if (saved_block_precision is not None) else None,
            )
            mixed_block_precision[i][name] = layer_block_precision
            quantizers['model.layers.%d.%s' % (i, name)] = (scales.cpu(), zeros.cpu(), g_idx.cpu(), args.low_quant_method, args.groupsize)

            gptq[name].free()


        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps
    # print("The average bit-width is:  ", sum(mean_bit_width) / len(mean_bit_width), " bits")
    if saved_block_precision is None:
        net = args.model.split("/")[-1]
        save_path = os.path.join(f'./block_precision_{args.groupsize}_{args.low_quant_method}/',f'{net}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(mixed_block_precision, save_path)
    model.config.use_cache = use_cache
    return quantizers

def pack_model(model, save_path, bits, group_size, quantizers, block_bits):
    # import sys
    # sys.path.append('/home/hw/SliM-LLM/AutoGPTQ')
    # print(sys.path)
    from auto_gptq import LlamaGPTQForCausalLM, OPTGPTQForCausalLM, BaseQuantizeConfig
    quantize_config = BaseQuantizeConfig(
        bits=bits,  # quantize model to 3-bit
        group_size=group_size,  # it is recommended to set the value to 128
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    )
    scales={k : quantizers[k][0] for k in quantizers}
    zeros={k : quantizers[k][1] for k in quantizers}
    g_idxes={k : quantizers[k][2] for k in quantizers}

    if "llama" in args.model:
        model = LlamaGPTQForCausalLM(model, quantized=False, quantize_config=quantize_config)
        model.quantize(None, scales, zeros, g_idxes, block_bits)
        model.save_quantized(save_path, use_safetensors=False)
    elif "opt" in args.model:
        model = OPTGPTQForCausalLM(model, quantized=False, quantize_config=quantize_config)
        model.quantize(None, scales, zeros, g_idxes, block_bits)
        model.save_quantized(save_path, use_safetensors=False)

if __name__ == "__main__":
    import argparse
    from datautils import *

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `huggyllama/llama-7b`."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "mix"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "low_quant_method",
        type=str,
        choices=["2bit", "3bit", "4bit", "5bit", "6bit", "7bit", "8bit"],
        help="weight bit-width",
    )
    parser.add_argument("--load_quantized", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=128,
        help="Group size to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="set the device to use for quantization.",
    )
    parser.add_argument(
        "--disable_gptq",
        action="store_true",
        help="disable GPTQ for quantization.",
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Quant all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Quant all layers with id < this."
    )
    parser.add_argument(
        "--quant_only",
        type=str,
        default="",
        help="Quant only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )

    parser.add_argument(
        "--salient_block",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--nonsalient_block",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--quantizer_matric",
        type=str,
        default="mse",
        help="quantizer parameter determination metric",
    )

    parser.add_argument(
        "--lambda_salience",
        type=float,
        default=1,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--tasks", 
        type=str,
        default="",
        help="Test datasets",
    )
    parser.add_argument(
        "--pack", action="store_true", help="Whether to save the packed model."
    )

    args = parser.parse_args()
    print(args)

    # get the block precision of the model
    # if the block precision does not exist, start Salience-Determined Bit Allocation
    groupsize = args.groupsize
    net = args.model.split("/")[-1]
    block_configurations = f'../SliM-LLM_group-precision/block_precision_{args.groupsize}_{args.low_quant_method}/{net}.pt'
    if os.path.exists(block_configurations):
        block_precision = torch.load(block_configurations)
    else:
        print(f'Block precisions of {net} does not exist. Start aware!')
        block_precision = None
    

    device = args.device
    save_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}"
    save_file = "./output/" + save_title.replace("/", "_") + ".pt"
    if args.load_quantized:
        model = get_model(save_file)
        model.eval()
    else:
        model = get_model(args.model)
        model.eval()
        tick = time.time()
        dataloader, testloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )
        quantizers = quant_sequential(model, dataloader, device, block_precision)
        print("quantization time:", time.time() - tick, "s")

    for dataset in ["wikitext2", "ptb", "c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, seqlen=model.seqlen, model=args.model
        )
        print(dataset)
        if "opt" in args.model:
            from eval_ppl_utils import opt_eval

            opt_eval(model, testloader, device, dataset, args.log_wandb)
        elif "llama" in args.model:
            from eval_ppl_utils import llama_eval

            llama_eval(model, testloader, device, dataset, args.log_wandb)
    if args.tasks != "":
        from eval_ppl_utils import zeroshot_evaluate
        zeroshot_evaluate(model, args)

    if args.save:
        # save the packed results with mixed-precision
        if args.pack:
            bits = int(args.low_quant_method[0])
            pack_model(model, save_file, bits, groupsize, quantizers, block_precision)
        # save the fake quantized resulta with FP14 format
        else:
            save_path = os.path.dirname(save_file)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model.save_pretrained(save_file)