import torch
from utils.reconstruct import error_computing, ssim, kl_div

index = 0

@torch.no_grad()
def binary(x):
    zero = torch.mean(x, dim=1)
    scale = torch.mean(torch.abs(x), dim=1)
    binary = torch.zeros_like(x)
    binary += x
    binary -= zero[:, None]
    binary_slice = torch.sign(binary) * scale[:, None]
    binary_slice += zero[:, None]
    return binary_slice

@torch.no_grad()
def normal_quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

@torch.no_grad()
def block_quantize(w, bit_width=0):
    if bit_width > 0:
        method = str(bit_width)+'bit'
    else:
        method = "prune"

    if method in ['2bit','4bit','3bit', '1bit', '5bit', '6bit', '7bit', '8bit', '9bit']:
        if method=="1bit":
            w = binary(w)
            return w
        # print("this block bit_width: ", method)
        bits = int(method[0])
        perchannel = True
        weight = True
        dev = w.device
        maxq = torch.tensor(2 ** bits - 1)
        scale = torch.zeros(1)
        zero = torch.zeros(1)
        scale=scale.to(dev)
        zero=zero.to(dev)
        maxq=maxq.to(dev)
        x = w.clone()
        shape = x.shape

        if perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)
        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
        scale = (xmax - xmin) / maxq
        zero = torch.round(-xmin / scale)

        if not perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            scale = scale.repeat(tmp)
            zero = zero.repeat(tmp)
        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            scale = scale.reshape(shape)
            zero = zero.reshape(shape)
        w = normal_quantize(w, scale, zero, maxq)
    return w

def block_search(salience_array, new_labels, w, columns, blocksize, bit_width):

    max_round = salience_array.shape[0] // 2
    best = float('inf')
    best_bits = []
    errors = []
    search_bits = []
    for i in range(1, max_round):
        tmp = torch.zeros_like(w)
        min_indices = salience_array.argsort()[:i]
        max_indices = salience_array.argsort()[-i:]
        tmp_bits = new_labels.copy()
        for j in min_indices:
            tmp_bits[j] = 0
        for j in max_indices:
            tmp_bits[j] = 2
        for blocki, col_st in enumerate(range(0, columns, blocksize)):
            col_ed = min(col_st + blocksize, columns)
            st = col_st
            ed = col_ed
            if tmp_bits[blocki] == 0:
                block_bit = bit_width - 1
            elif tmp_bits[blocki] == 2:
                block_bit = bit_width + 1
            else:
                block_bit = bit_width
            search_bits.append(block_bit)
            tmp[:, st:ed] = block_quantize(w[:, st:ed], bit_width=block_bit)
        
        error = error_computing(tmp, w)
        errors.append(error.item())
        print(search_bits, error.item())
        search_bits = []
        if error < best:
            best_bits = tmp_bits.copy()
    
    import matplotlib.pyplot as plt

    global index
    index += 1
    plt.figure()
    plt.plot(errors)
    plt.title('Error Curve')
    plt.xlabel('Mixed Blocks')
    plt.ylabel('Error')
    plt.show()
    plt.savefig('plt_block_error/' + str(index) + '.png')
    exit()        
    # print(best_bits)
    return best_bits


def activation_aware_search(salience_array, new_labels, w, columns, blocksize, bit_width, activation_in, activation_out):

    max_round = salience_array.shape[0] // 2

    errors = []
    search_bits = []
    for i in range(1, max_round):
        tmp = torch.zeros_like(w)
        min_indices = salience_array.argsort()[:i]
        max_indices = salience_array.argsort()[-i:]
        tmp_bits = new_labels.copy()
        for j in min_indices:
            tmp_bits[j] = 0
        for j in max_indices:
            tmp_bits[j] = 2
        for blocki, col_st in enumerate(range(0, columns, blocksize)):
            col_ed = min(col_st + blocksize, columns)
            st = col_st
            ed = col_ed
            if tmp_bits[blocki] == 0:
                block_bit = bit_width - 1
            elif tmp_bits[blocki] == 2:
                block_bit = bit_width + 1
            else:
                block_bit = bit_width
            search_bits.append(block_bit)
            tmp[:, st:ed] = block_quantize(w[:, st:ed], bit_width=block_bit)
        
        error = kl_div(activation_in@tmp.T, activation_out)
        errors.append(error.item())

    return errors



