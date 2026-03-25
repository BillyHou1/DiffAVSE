#Author: Dominic and Zhenning
#Mamba blocks for T-F spectrogram denoising
#CausalTFMambaBlock: modified Mamba to causal in time, bidirectional in frequency.
import torch
import torch.nn as nn
from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba
try:
    from mamba_ssm.modules.mamba_simple import Block
except ImportError:
    from mamba_ssm.modules.block import Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm

def _getMixerCls(cfg,layer_idx=0):
    d_state=cfg['model_cfg']['d_state']
    d_conv=cfg['model_cfg']['d_conv']
    expand=cfg['model_cfg']['expand']
    return partial(Mamba,layer_idx=layer_idx,d_state=d_state,d_conv=d_conv,expand=expand)

def create_block(d_model,cfg,layer_idx=0,rms_norm=True,
                 fused_add_norm=False,residual_in_fp32=False):
    normEps=cfg['model_cfg']['norm_epsilon']
    mixerCls=_getMixerCls(cfg,layer_idx)
    normCls=partial(nn.LayerNorm if not rms_norm else RMSNorm,eps=normEps)
    block=Block(d_model,mixerCls,mlp_cls=nn.Identity,
                norm_cls=normCls,fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32)
    block.layer_idx=layer_idx
    return block

class MambaBlock(nn.Module):
    def __init__(self,in_channels,cfg,bidirectional=True):
        super(MambaBlock,self).__init__()
        n_layer=1
        self.bidirectional=bidirectional
        self.forward_blocks=nn.ModuleList(create_block(in_channels,cfg) for i in range(n_layer))
        if bidirectional:
            self.backward_blocks=nn.ModuleList(create_block(in_channels,cfg) for i in range(n_layer))
        self.apply(partial(_init_weights,n_layer=n_layer))

    def forward(self,x):
        xf=x.clone()
        rf=None
        for layer in self.forward_blocks:
            xf,rf=layer(xf,rf)
        yf=(xf+rf) if rf is not None else xf
        if not self.bidirectional:
            return yf
        xb=torch.flip(x,[1])
        rb=None
        for layer in self.backward_blocks:
            xb,rb=layer(xb,rb)
        yb=torch.flip((xb+rb),[1]) if rb is not None else torch.flip(xb,[1])
        return torch.cat([yf,yb],-1)

#original SEMamba is bidirectional in both time and freq
class TFMambaBlock(nn.Module):
    def __init__(self,cfg):
        super(TFMambaBlock,self).__init__()
        self.cfg=cfg
        self.hid_feature=cfg['model_cfg']['hid_feature']
        self.time_mamba=MambaBlock(in_channels=self.hid_feature,cfg=cfg)
        self.freq_mamba=MambaBlock(in_channels=self.hid_feature,cfg=cfg)
        self.tlinear=nn.ConvTranspose1d(self.hid_feature*2,self.hid_feature,1,stride=1)
        self.flinear=nn.ConvTranspose1d(self.hid_feature*2,self.hid_feature,1,stride=1)

    def forward(self,x):
        b,c,t,f=x.size()
        x=x.permute(0,3,2,1).contiguous().view(b*f,t,c)
        x=self.tlinear(self.time_mamba(x).permute(0,2,1)).permute(0,2,1)+x
        x=x.view(b,f,t,c).permute(0,2,1,3).contiguous().view(b*t,f,c)
        x=self.flinear(self.freq_mamba(x).permute(0,2,1)).permute(0,2,1)+x
        x=x.view(b,t,f,c).permute(0,3,1,2)
        return x

#LiteAVSEMamba is a causal in time,bidirectional in freq
class CausalTFMambaBlock(nn.Module):
    def __init__(self,cfg):
        super(CausalTFMambaBlock,self).__init__()
        self.hid_feature=cfg['model_cfg']['hid_feature']
        self.time_mamba=MambaBlock(in_channels=self.hid_feature,cfg=cfg,bidirectional=False)
        self.freq_mamba=MambaBlock(in_channels=self.hid_feature,cfg=cfg,bidirectional=True)
        #time is causal(C->C),freq is bidirectional(2C->C)
        self.tlinear=nn.ConvTranspose1d(self.hid_feature,self.hid_feature,1,stride=1)
        self.flinear=nn.ConvTranspose1d(self.hid_feature*2,self.hid_feature,1,stride=1)

    def forward(self,x):
        b,c,t,f=x.size()
        x=x.permute(0,3,2,1).contiguous().view(b*f,t,c)
        x=self.tlinear(self.time_mamba(x).permute(0,2,1)).permute(0,2,1)+x
        x=x.view(b,f,t,c).permute(0,2,1,3).contiguous().view(b*t,f,c)
        x=self.flinear(self.freq_mamba(x).permute(0,2,1)).permute(0,2,1)+x
        x=x.view(b,t,f,c).permute(0,3,1,2)
        return x
