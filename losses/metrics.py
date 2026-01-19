import torch

def rmse(pred, gt):
    mask = gt > 0
    return torch.sqrt(((pred[mask] - gt[mask]) ** 2).mean())