CUDA_VISIBLE_DEVICES=0 python main.py \
--model ../Model/llama2/Llama-2-7b-hf  --eval_ppl \
--epochs 50 --output_dir ./log/llama-2-7b-w2a16g128 \
--wbits 2 --abits 16 --group_size 128 --lwc --aug_loss

CUDA_VISIBLE_DEVICES=0 python main.py \
--model ../Model/llama2/Llama-2-7b-hf  --eval_ppl \
--epochs 50 --output_dir ./log/llama-2-7b-w3a16g128 \
--wbits 3 --abits 16 --group_size 128 --lwc --aug_loss

CUDA_VISIBLE_DEVICES=0 python main.py \
--model ../Model/llama2/Llama-2-13b-hf  --eval_ppl \
--epochs 50 --output_dir ./log/llama-2-13b-w2a16g128 \
--wbits 2 --abits 16 --group_size 128 --lwc --aug_loss

CUDA_VISIBLE_DEVICES=0 python main.py \
--model ../Model/llama2/Llama-2-13b-hf  --eval_ppl \
--epochs 50 --output_dir ./log/llama-2-13b-w3a16g128 \
--wbits 3 --abits 16 --group_size 128 --lwc --aug_loss

CUDA_VISIBLE_DEVICES=0 python main.py \
--model ../Model/llama2/Llama-2-70b-hf  --eval_ppl \
--epochs 50 --output_dir ./log/llama-2-170b-w2a16g128 \
--wbits 2 --abits 16 --group_size 128 --lwc --aug_loss

CUDA_VISIBLE_DEVICES=0 python main.py \
--model ../Model/llama2/Llama-2-70b-hf  --eval_ppl \
--epochs 50 --output_dir ./log/llama-2-70b-w3a16g128 \
--wbits 3 --abits 16 --group_size 128 --lwc --aug_loss