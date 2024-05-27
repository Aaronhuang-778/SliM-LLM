import torch
import torch.nn.functional as F


@torch.no_grad()
def error_computing(quantized_matrix, origin_matrix):
    mse = torch.mean((origin_matrix - quantized_matrix) ** 2)
    print(origin_matrix.shape, quantized_matrix.shape, mse)
    return mse

@torch.no_grad()
def kl_div(quantized_matrix, origin_matrix):
    tensor1 = F.softmax(quantized_matrix, dim=-1)
    tensor2 = F.softmax(origin_matrix, dim=-1)
    
    tensor1 = tensor1.clamp(min=1e-6)
    tensor2 = tensor2.clamp(min=1e-6)
    kl_div = F.kl_div(torch.log(tensor2), tensor1, reduction='batchmean')
    return kl_div

@torch.no_grad()
def ssim(x, y, C1=0.01**2, C2=0.03**2):
    n, m = x.shape
    
    # 计算平均值
    mu_x = x.mean(dim=1, keepdim=True)
    mu_y = y.mean(dim=1, keepdim=True)
    
    # 计算方差
    sigma_x = x.var(dim=1, unbiased=False, keepdim=True)
    sigma_y = y.var(dim=1, unbiased=False, keepdim=True)
    
    # 计算协方差
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=1, keepdim=True)
    
    # 计算SSIM
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2)
    ssim_vals = numerator / denominator
  
    return ssim_vals.mean()