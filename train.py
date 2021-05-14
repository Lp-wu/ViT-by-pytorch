import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models import VisionTransformer


v = VisionTransformer(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    Dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(2, 3, 256, 256)

preds = v(img) # (1, 1000)
