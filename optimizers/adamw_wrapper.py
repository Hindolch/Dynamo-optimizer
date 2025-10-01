import torch
from torch.optim import AdamW

def AdamWWrapper(params, **kwargs):
    return AdamW(params, **kwargs)
