import torch

from torch import nn
from einops import rearrange


class Involution(nn.Module):
    def __init__(self, channel_num, group_num=1, kernel_size=3, stride=1, reduction_ratio=2):
        super().__init__()
        self.o = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()
        self.reduce = nn.Conv2d(channel_num, channel_num // reduction_ratio, 1)
        self.bn = nn.BatchNorm2d(channel_num // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.span = nn.Conv2d(channel_num // reduction_ratio, kernel_size * kernel_size * group_num, 1)
        self.unfold = nn.Unfold(kernel_size, padding=kernel_size // 2, stride=stride)
        self.k = kernel_size
        
    def forward(self, x):
        kernel = rearrange(self.span(self.relu(self.bn(self.reduce(self.o(x))))), 'b (k j g) h w -> b g (k j) h w', k=self.k, j=self.k)
        b, g, _, h, w = kernel.size()
        x = rearrange(self.unfold(x), 'b (g d k j) (h w) -> b g d (k j) h w', g=g, k=self.k, j=self.k, h=h, w=w)
        out = rearrange(torch.einsum('bgdxhw, bgxhw -> bgdhw', x, kernel), 'b g d h w -> b (g d) h w')
        return out
