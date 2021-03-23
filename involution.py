import torch

from torch import nn
from einops import rearrange


class Involution(nn.Module):
    def __init__(self, channels, groups=1, kernel_size=3, stride=1, reduction_ratio=2):
        super().__init__()
        
        channels_reduced = channels // reduction_ratio
        padding = kernel_size // 2
        
        self.o = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()
        
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, channels_reduced, 1),
            nn.BatchNorm2d(channels_reduced),
            nn.ReLU(inplace=True))
        
        self.span = nn.Conv2d(channels_reduced, kernel_size * kernel_size * groups, 1)
        self.unfold = nn.Unfold(kernel_size, padding=padding, stride=stride)
        
        self.k = kernel_size
        
    def forward(self, x):
        kernel = rearrange(self.span(self.reduce(self.o(x))), 'b (k j g) h w -> b g (k j) h w', k=self.k, j=self.k)
        _, g, _, h, w = kernel.size()
        x = rearrange(self.unfold(x), 'b (g d k j) (h w) -> b g d (k j) h w', g=g, k=self.k, j=self.k, h=h, w=w)
        out = rearrange(torch.einsum('bgdxhw, bgxhw -> bgdhw', x, kernel), 'b g d h w -> b (g d) h w')
        return out
