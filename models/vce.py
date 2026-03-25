#author:Billy
#Visual Confidence Estimator, scores each frame's reliability as alpha in[0,1]
#two variants
#1. CrossModalVCE: uses both visual+audio context(ScoutNeed)
#2. VCEWithTemporalSmoothing: visualOnly(simple VCE+FSVG baseline)
import torch
import torch.nn as nn
import torch.nn.functional as F
class CrossModalVCE(nn.Module):
    #[B,T,D_vis]+[B,T,D_aud]->alpha[B,T,1]
    def __init__(self,visual_dim=512,audio_dim=64,hidden_dim=128,smooth_kernel=5):
        super().__init__()
        self.visualProj=nn.Linear(visual_dim,hidden_dim)
        self.audioProj=nn.Linear(audio_dim,hidden_dim)
        self.scoreNet=nn.Sequential(
            nn.Linear(hidden_dim*2,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),
            nn.Sigmoid())
        self.smoothKernel=smooth_kernel
        if smooth_kernel>1:
            self.smoother=nn.Conv1d(1,1,kernel_size=smooth_kernel,padding=0,bias=False)
            nn.init.constant_(self.smoother.weight,1.0/smooth_kernel)
        else:
            self.smoother=None

    def forward(self,visual_feat,audio_feat):
        v=self.visualProj(visual_feat)
        a=self.audioProj(audio_feat)
        alpha=self.scoreNet(torch.cat([v,a],dim=-1))
        if self.smoother is not None:
            alpha=alpha.permute(0,2,1)
            alpha=F.pad(alpha,(self.smoothKernel-1,0))
            alpha=self.smoother(alpha)
            alpha=alpha.permute(0,2,1)
            alpha=torch.clamp(alpha,0.0,1.0)
        return alpha

class VCEWithTemporalSmoothing(nn.Module):
    #[B,T,D_vis]->alpha[B,T,1](visualonly, there no audio context)
    def __init__(self,input_dim=512,hidden_dims=None,smooth_kernel=5):
        super().__init__()
        if hidden_dims is None:
            hidden_dims=[256,64]
        layers=[]
        dims=[input_dim]+hidden_dims
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i],dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1],1))
        layers.append(nn.Sigmoid())
        self.net=nn.Sequential(*layers)
        self.smoothKernel=smooth_kernel
        self.smoother=nn.Conv1d(1,1,kernel_size=smooth_kernel,padding=0,bias=False)
        nn.init.constant_(self.smoother.weight,1.0/smooth_kernel)

    def forward(self,visual_feat):
        alpha=self.net(visual_feat)
        alpha=alpha.permute(0,2,1)
        alpha=F.pad(alpha,(self.smoothKernel-1,0))
        alpha=self.smoother(alpha)
        alpha=alpha.permute(0,2,1)
        alpha=torch.clamp(alpha,0.0,1.0)
        return alpha
