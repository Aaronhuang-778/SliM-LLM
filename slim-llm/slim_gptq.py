import math
import time
import torch
import torch.nn as nn
import transformers
import numpy as np
from utils.bitsearch import activation_aware_search


DEBUG = False
index = 0

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

'''
SliM-LLM implementation of GPTQ
'''

class SliMGPTQ:
    def __init__(
        self, layer, quantizer, disable_gptq=False, layer_index=0, salient_block=-1, nonsalient_block=-1, 
        bit_width=2):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = quantizer
        self.disable_gptq = disable_gptq
        self.layer_index = layer_index
        self.salient_block = salient_block
        self.nonsalient_block = nonsalient_block
        self.bit_width = bit_width
        self.block_errors = []
        self.block_salience = []

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
    
    def get_salience(self, blocksize=128):
        h = self.H
        w = self.layer.weight.data.clone()
        dead = torch.diag(h) == 0
        h[dead, dead] = 1
        diag = torch.arange(self.columns, device=self.dev)
        damp = 0.01
        h[diag, diag] += damp
        h = torch.linalg.cholesky(h)
        h = torch.cholesky_inverse(h)
        h = torch.linalg.cholesky(h, upper=True)
        Hinv = h
        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            st = col_st
            ed = col_ed
            block_value = w[:, st:ed] ** 2 / (torch.diag(Hinv[st:ed, st:ed]).reshape((1, -1))) ** 2
            self.block_salience.append(torch.sum(block_value).item())


    def get_block(self, inp, out, blocksize=128):
        W = self.layer.weight.data.clone()
        new_labels = [1] * len(self.block_salience)
        salience_array = np.array(self.block_salience)
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        error = activation_aware_search(salience_array, new_labels, W, self.columns, blocksize, self.bit_width, inp, out)
        self.block_errors.append(error)
        del W

    def fasterquant(self,
                    blocksize=128, 
                    percdamp=0.01, 
                    layer_name = "",
                    saved_block_precision = None,
                    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        block_precision = []

        if self.salient_block != -1:
            high_bit_num = self.salient_block
            low_bit_num = self.nonsalient_block
            block_salience = []
            for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
                col_ed = min(col_st + blocksize, self.columns)
                n_cols = col_ed - col_st
                st = col_st
                ed = col_ed
                block_value = W[:, st:ed] ** 2 / (torch.diag(Hinv[st:ed, st:ed]).reshape((1, -1))) ** 2
                block_salience.append(torch.sum(block_value).item())
            new_labels = [1] * len(block_salience)
            salience_array = np.array(block_salience)

            min_indices = salience_array.argsort()[:low_bit_num]
            max_indices = salience_array.argsort()[-high_bit_num:]
            for i in min_indices:
                new_labels[i] = 0
            for i in max_indices:
                new_labels[i] = 2
        else: 
            if saved_block_precision is None:

                zipped = zip(*self.block_errors)
                average_error = [np.mean(item) for item in zipped]
                high_bit_num = average_error.index(min(average_error)) + 1
                low_bit_num = high_bit_num

                block_salience = []
                for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
                    col_ed = min(col_st + blocksize, self.columns)
                    n_cols = col_ed - col_st
                    st = col_st
                    ed = col_ed
                    block_value = W[:, st:ed] ** 2 / (torch.diag(Hinv[st:ed, st:ed]).reshape((1, -1))) ** 2
                    block_salience.append(torch.sum(block_value).item())
                new_labels = [1] * len(block_salience)
                salience_array = np.array(block_salience)

                min_indices = salience_array.argsort()[:low_bit_num]
                max_indices = salience_array.argsort()[-high_bit_num:]
                for i in min_indices:
                    new_labels[i] = 0
                for i in max_indices:
                    new_labels[i] = 2
            else:
                block_precision = saved_block_precision

        g_idx = [i // blocksize for i in range(self.columns)]
        scales = torch.zeros((W.shape[0], g_idx[-1] + 1), device=self.dev)
        zeros = torch.zeros((W.shape[0], g_idx[-1] + 1), device=self.dev)
        g_idx = torch.tensor(g_idx, dtype=torch.int32)
        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            n_cols = col_ed - col_st
            st = col_st
            ed = col_ed
            block_bit = 0

            assert self.quantizer.groupsize % blocksize == 0

            if self.disable_gptq:
                w = W[:, col_st:col_ed]
                self.quantizer.sym = False

                self.quantizer.fit(W1,  bit_width=self.bit_width)
                W[:, col_st:col_ed] = self.quantizer.quantize(w, bit_width=self.bit_width)
            else:
                # shape of W1: [oc, n_cols]
                W1 = W[:, col_st:col_ed].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

                if self.salient_block != -1 or saved_block_precision is None:
                    if new_labels[blocki] == 0:
                        block_bit = self.bit_width - 1
                    elif new_labels[blocki] == 2:
                        block_bit = self.bit_width + 1
                        self.quantizer.sym = False
                    else:
                        block_bit = self.bit_width
                        self.quantizer.sym = False
                    block_precision.append(block_bit)
                # print(str(bit_width) + "-bit")
                else:
                    block_bit = block_precision[blocki]
                    self.quantizer.sym = False

                self.quantizer.fit(W1, Hinv1, bit_width=block_bit)
                zeros[:, g_idx[col_st]] = self.quantizer.zero.squeeze()
                scales[:, g_idx[col_st]] = self.quantizer.scale.squeeze()
                for i in range(n_cols):
                    # shape of w: [oc, 1]
                    wi = W1[:, i]
                    d = Hinv1[i, i]
                    qi = self.quantizer.quantize(wi, bit_width=block_bit)
                    Q1[:, i] = qi
                    Losses1[:, i] = (wi - qi) ** 2 / d**2
                    # breakpoint()
                    err1 = (wi - qi) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                W[:, col_st:col_ed] = Q1
                Losses += torch.sum(Losses1, 1) / 2
                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

                if DEBUG:
                    self.layer.weight.data[:, :col_ed] = W[:, :col_ed]
                    self.layer.weight.data[:, col_ed:] = W[:, col_ed:]
                    print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                    print(torch.sum(Losses))
            self.quantizer.clear_quantize_paremeter()

        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - tick))
        print("error", torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        if not self.disable_gptq:
            del W1, Q1, W, Err1, Losses1, Hinv1
        del H, Hinv
        torch.cuda.empty_cache()
        print("block precisions: ", block_precision)
        return block_precision, scales, zeros, g_idx

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
