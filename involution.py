import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class Involution(nn.Module):
    """
    Implementation of `Involution: Inverting the Inherence of Convolution for Visual Recognition`.
    """
    def __init__(self, in_channels, out_channels, groups=1, kernel_size=3, stride=1, reduction_ratio=2):

        super().__init__()

        channels_reduced = max(1, in_channels // reduction_ratio)
        padding = kernel_size // 2

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, channels_reduced, 1),
            nn.BatchNorm2d(channels_reduced),
            nn.ReLU(inplace=True))

        self.span = nn.Conv2d(channels_reduced, kernel_size * kernel_size * groups, 1)
        self.unfold = nn.Unfold(kernel_size, padding=padding, stride=stride)
        
        self.resampling = None if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

    @classmethod
    def get_name(cls):
        """
        Return this layer name.

        Returns:
            str: layer name.
        """
        return 'Involution'

    def forward(self, input_tensor):
        """
        Calculate Involution.

        override function from PyTorch.
        """
        _, _, height, width = input_tensor.size()
        if self.stride > 1:
            out_size = lambda x: (x + 2 * self.padding - self.kernel_size) // self.stride + 1
            height, width = out_size(height), out_size(width)
        uf_x = rearrange(self.unfold(input_tensor), 'b (g d k j) (h w) -> b g d (k j) h w',
                         g=self.groups, k=self.kernel_size, j=self.kernel_size, h=height, w=width)

        if self.stride > 1:
            input_tensor = F.adaptive_avg_pool2d(input_tensor, (height, width))
        kernel = rearrange(self.span(self.reduce(input_tensor)), 'b (k j g) h w -> b g (k j) h w',
                           k=self.kernel_size, j=self.kernel_size)

        out = rearrange(torch.einsum('bgdxhw, bgxhw -> bgdhw', uf_x, kernel), 'b g d h w -> b (g d) h w')
        
        if self.resampling:
            out = self.resampling(out)
            
        return out