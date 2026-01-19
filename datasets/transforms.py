import torch

def to_tensor(img, depth):
    img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    depth = torch.from_numpy(depth).float()
    return img, depth