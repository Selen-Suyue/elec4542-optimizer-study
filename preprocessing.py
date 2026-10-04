"""Shared input transform; use independent generators for order and augmentation."""
import torch
from torch.nn import functional as F
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def preprocess(x, training, generator):
    x = x.float().div_(255)
    b = x.shape[0]
    if training:
        size = torch.sqrt(0.6 + 0.4 * torch.rand(b, device=x.device, generator=generator))
        dx = (2 * torch.rand(b, device=x.device, generator=generator) - 1) * (1 - size)
        dy = (2 * torch.rand(b, device=x.device, generator=generator) - 1) * (1 - size)
        flip = torch.where(torch.rand(b, device=x.device, generator=generator) < 0.5, -1.0, 1.0)
        theta = torch.zeros(b, 2, 3, device=x.device)
        theta[:, 0, 0] = size * flip
        theta[:, 1, 1] = size
        theta[:, 0, 2] = dx
        theta[:, 1, 2] = dy
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
    mean = torch.tensor(MEAN, device=x.device)[None, :, None, None]
    std = torch.tensor(STD, device=x.device)[None, :, None, None]
    return (x - mean) / std
