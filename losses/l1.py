def l1_loss(pred, gt):
    mask = gt > 0
    return (pred[mask] - gt[mask]).abs().mean()