import torch

def silog_loss(pred, gt):
    mask = gt > 0
    d = torch.log(pred[mask] + 1e-6) - torch.log(gt[mask])
    return torch.sqrt((d**2).mean() - 0.85 * (d.mean()**2))