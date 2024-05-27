
import torch

def generate_mask(origin_matrix, min_salience):
    
    mask0 = origin_matrix < min_salience
    mask1 = origin_matrix >= min_salience

    return mask0, mask1

def saliency_mask(hessian_matrix):
    threshold = 2

    mean = torch.mean(hessian_matrix)
    std_dev = torch.std(hessian_matrix)
    z_scores = (hessian_matrix - mean) / std_dev
    outliers = hessian_matrix[torch.abs(z_scores) > threshold]
    ourlier_min_values = torch.min(outliers)


    return generate_mask(hessian_matrix, ourlier_min_values)