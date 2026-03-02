# Zhenning
# DenseEncoder, MagDecoder, PhaseDecoder from MP-SENet are already done below.
# Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/models/generator.py
# Lite variants are at the bottom, Zhenning's job.
# Use them when cfg['lite_cfg']['use_lite_dense'] is True.

import torch
import torch.nn as nn
from einops import rearrange
from .lsigmoid import LearnableSigmoid2D

def get_padding(kernel_size, dilation=1):
    """
    Calculate the padding size for a convolutional layer.
    
    Args:
    - kernel_size (int): Size of the convolutional kernel.
    - dilation (int, optional): Dilation rate of the convolution. Defaults to 1.
    
    Returns:
    - int: Calculated padding size.
    """
    return int((kernel_size * dilation - dilation) / 2)

def get_padding_2d(kernel_size, dilation=(1, 1)):
    """
    Calculate the padding size for a 2D convolutional layer.
    
    Args:
    - kernel_size (tuple): Size of the convolutional kernel (height, width).
    - dilation (tuple, optional): Dilation rate of the convolution (height, width). Defaults to (1, 1).
    
    Returns:
    - tuple: Calculated padding size (height, width).
    """
    return (int((kernel_size[0] * dilation[0] - dilation[0]) / 2), 
            int((kernel_size[1] * dilation[1] - dilation[1]) / 2))

class DenseBlock(nn.Module):
    """
    DenseBlock module consisting of multiple convolutional layers with dilation.
    """
    def __init__(self, cfg, kernel_size=(3, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.cfg = cfg
        self.depth = depth
        self.dense_block = nn.ModuleList()
        self.hid_feature = cfg['model_cfg']['hid_feature']

        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(self.hid_feature * (i + 1), self.hid_feature, kernel_size, 
                          dilation=(dil, 1), padding=get_padding_2d(kernel_size, (dil, 1))),
                nn.InstanceNorm2d(self.hid_feature, affine=True),
                nn.PReLU(self.hid_feature)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        """
        Forward pass for the DenseBlock module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Output tensor after processing through the dense block.
        """
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x

class DenseEncoder(nn.Module):
    """
    DenseEncoder module consisting of initial convolution, dense block, and a final convolution.
    """
    def __init__(self, cfg):
        super(DenseEncoder, self).__init__()
        self.cfg = cfg
        self.input_channel = cfg['model_cfg']['input_channel']
        self.hid_feature = cfg['model_cfg']['hid_feature']

        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(self.input_channel, self.hid_feature, (1, 1)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )

        self.dense_block = DenseBlock(cfg, depth=4)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )

    def forward(self, x):
        """
        Forward pass for the DenseEncoder module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Encoded tensor.
        """
        x = self.dense_conv_1(x)  # [batch, hid_feature, time, freq]
        x = self.dense_block(x)   # [batch, hid_feature, time, freq]
        x = self.dense_conv_2(x)  # [batch, hid_feature, time, freq//2]
        return x

class MagDecoder(nn.Module):
    """
    MagDecoder module for decoding magnitude information.
    """
    def __init__(self, cfg):
        super(MagDecoder, self).__init__()
        self.dense_block = DenseBlock(cfg, depth=4)
        self.hid_feature = cfg['model_cfg']['hid_feature']
        self.output_channel = cfg['model_cfg']['output_channel']
        self.n_fft = cfg['stft_cfg']['n_fft']
        self.beta = cfg['model_cfg']['beta']

        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.Conv2d(self.hid_feature, self.output_channel, (1, 1)),
            nn.InstanceNorm2d(self.output_channel, affine=True),
            nn.PReLU(self.output_channel),
            nn.Conv2d(self.output_channel, self.output_channel, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid2D(self.n_fft // 2 + 1, beta=self.beta)

    def forward(self, x):
        """
        Forward pass for the MagDecoder module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Decoded tensor with magnitude information.
        """
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = rearrange(x, 'b c t f -> b f t c').squeeze(-1)
        x = self.lsigmoid(x)
        x = rearrange(x, 'b f t -> b t f').unsqueeze(1)
        return x

class PhaseDecoder(nn.Module):
    """
    PhaseDecoder module for decoding phase information.
    """
    def __init__(self, cfg):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(cfg, depth=4)
        self.hid_feature = cfg['model_cfg']['hid_feature']
        self.output_channel = cfg['model_cfg']['output_channel']

        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )

        self.phase_conv_r = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))

    def forward(self, x):
        """
        Forward pass for the PhaseDecoder module.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Decoded tensor with phase information.
        """
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x


# Lite variants - Zhenning
# Same structure as above but swap Conv2d for depthwise separable conv to cut params.
# Look up depthwise separable convolution if you haven't seen it,
# the key is the groups param in nn.Conv2d.
# Used when cfg['lite_cfg']['use_lite_dense'] is True.

class DepthwiseSeparableConv2d(nn.Module):
    """
    Drop-in replacement for nn.Conv2d using depthwise separable convolution.
    Same interface as nn.Conv2d (in_channels, out_channels, kernel_size, ...).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        super().__init__()
        # TODO depthwise separable conv, same args as nn.Conv2d
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class LiteDenseBlock(nn.Module):
    """
    Same structure as DenseBlock but uses DepthwiseSeparableConv2d
    instead of nn.Conv2d. Everything else (norm, activation, skip connections)
    stays the same.
    """
    def __init__(self, cfg, kernel_size=(3, 3), depth=4):
        super().__init__()
        self.depth = depth
        # TODO same as DenseBlock but use DepthwiseSeparableConv2d
        raise NotImplementedError

    def forward(self, x):
        """Same forward logic as DenseBlock."""
        raise NotImplementedError


class LiteDenseEncoder(nn.Module):
    """
    Same as DenseEncoder but uses LiteDenseBlock.
    The 1x1 and small convs (dense_conv_1, dense_conv_2) stay standard Conv2d.
    """
    def __init__(self, cfg):
        super().__init__()
        # TODO copy DenseEncoder, swap DenseBlock for LiteDenseBlock
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class LiteMagDecoder(nn.Module):
    """Same as MagDecoder but uses LiteDenseBlock."""
    def __init__(self, cfg):
        super().__init__()
        # TODO copy MagDecoder, swap DenseBlock for LiteDenseBlock
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class LitePhaseDecoder(nn.Module):
    """Same as PhaseDecoder but uses LiteDenseBlock."""
    def __init__(self, cfg):
        super().__init__()
        # TODO copy PhaseDecoder, swap DenseBlock for LiteDenseBlock
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
