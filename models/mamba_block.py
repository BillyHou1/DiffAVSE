# Author: Zhenning
# Mamba blocks for time-frequency scanning. CausalTFMambaBlock is the main one.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from functools import partial
from einops import rearrange

# mamba-ssm 2.x changed import paths, try new first then fall back
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.modules.block import Block
    from mamba_ssm.models.mixer_seq_simple import _init_weights
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    from mamba_ssm.modules.mamba_simple import Mamba, Block
    from mamba_ssm.models.mixer_seq_simple import _init_weights
    from mamba_ssm.ops.triton.layernorm import RMSNorm


def create_block(
    d_model, cfg, layer_idx=0, rms_norm=True, fused_add_norm=False, residual_in_fp32=False,
    ):
    d_state = cfg['model_cfg']['d_state'] # 16
    d_conv = cfg['model_cfg']['d_conv'] # 4
    expand = cfg['model_cfg']['expand'] # 4
    norm_epsilon = cfg['model_cfg']['norm_epsilon'] # 0.00001

    #TODO swap this with _get_mixer_cls(cfg, layer_idx) once you have it
    mixer_cls = partial(Mamba, layer_idx=layer_idx, d_state=d_state, d_conv=d_conv, expand=expand)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon
    )
    #TODO on mamba-ssm 2.x you need mlp_cls=nn.Identity here
    # 1.2.x doesn't accept that arg so don't add it if you're on the bundled version
    block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            )
    block.layer_idx = layer_idx
    return block


class MambaBlock(nn.Module):
    def __init__(self, in_channels, cfg, bidirectional=True):
        super(MambaBlock, self).__init__()
        self.bidirectional = bidirectional
        n_layer = 1
        self.forward_blocks = nn.ModuleList(create_block(in_channels, cfg) for i in range(n_layer))
        if self.bidirectional:
            self.backward_blocks = nn.ModuleList(create_block(in_channels, cfg) for i in range(n_layer))

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
            )
        )

    def forward(self, x):
        x_forward = x.clone()
        resi_forward = None

        for layer in self.forward_blocks:
            x_forward, resi_forward = layer(x_forward, resi_forward)
        y_forward = (x_forward + resi_forward) if resi_forward is not None else x_forward

        if not self.bidirectional:
            return y_forward  # [B, T, C]

        x_backward = torch.flip(x, [1])
        resi_backward = None
        for layer in self.backward_blocks:
            x_backward, resi_backward = layer(x_backward, resi_backward)
        y_backward = torch.flip((x_backward + resi_backward), [1]) if resi_backward is not None else torch.flip(x_backward, [1])

        return torch.cat([y_forward, y_backward], -1)  # [B, T, 2C]


class TFMambaBlock(nn.Module):
    """Original bidirectional TF-Mamba (non-causal, for ablation)."""
    def __init__(self, cfg):
        super(TFMambaBlock, self).__init__()
        self.cfg = cfg
        self.hid_feature = cfg['model_cfg']['hid_feature']

        self.time_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg)
        self.freq_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg)

        # both are 2C -> C cause both mamba blocks are bidirectional so they output 2C
        self.tlinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)

    def forward(self, x):
        b, c, t, f = x.size()

        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.tlinear( self.time_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.flinear( self.freq_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class CausalTFMambaBlock(nn.Module):
    """Same as TFMambaBlock but time is causal (forward-only), freq stays bidirectional."""
    def __init__(self, cfg):
        super(CausalTFMambaBlock, self).__init__()
        self.cfg = cfg
        self.hid_feature = cfg['model_cfg']['hid_feature']

        self.time_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg, bidirectional=False)
        self.freq_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg, bidirectional=True)

        # time causal -> output C, freq bidir -> output 2C
        self.tlinear = nn.ConvTranspose1d(self.hid_feature, self.hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)

    def forward(self, x):
        b, c, t, f = x.size()

        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.tlinear(self.time_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.flinear(self.freq_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x
