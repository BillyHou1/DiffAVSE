#Dominic part
#FSVG gates how much visual info gets injected at each freq bin
#lip movements help most with speech frequencies 300Hz-3kHz
import torch
import torch.nn as nn

class FSVG(nn.Module):
    #GMU-based gating, context kernel smooths gate across freq bins
    def __init__(self,in_channels=None,mode="basic",hidden_channels=None,
                 context_kernel=3,use_interactions=True,audio_channels=None,
                 visual_channels=None,alpha_channels=0,use_energy_saliency=False):
        super().__init__()
        if in_channels is None:
            in_channels=audio_channels
        if in_channels is None:
            raise ValueError("Must provide in_channels or audio_channels")
        if hidden_channels is None:
            hidden_channels=max(8,in_channels//2)
        if context_kernel%2==0:
            raise ValueError("context_kernel must be odd")

        self.mode=str(mode).lower()
        self.useInteractions=bool(use_interactions)
        self.alphaChannels=alpha_channels
        self.useEnergySaliency=use_energy_saliency

        salCh=1 if use_energy_saliency else 0
        if self.mode=="basic" or not self.useInteractions:
            fusionCh=2*in_channels+alpha_channels+salCh
        else:
            fusionCh=4*in_channels+alpha_channels+salCh

        if self.mode=="basic":
            self.gateNet=nn.Sequential(
                nn.Conv2d(fusionCh,hidden_channels,kernel_size=1,bias=True),
                nn.SiLU(),
                nn.Conv2d(hidden_channels,1,kernel_size=1,bias=True),
            )
        elif self.mode=="enhanced":
            self.gateNet=nn.Sequential(
                nn.Conv2d(fusionCh,hidden_channels,kernel_size=1,bias=True),
                nn.SiLU(),
                nn.Conv2d(hidden_channels,hidden_channels,kernel_size=context_kernel,
                    padding=context_kernel//2,groups=hidden_channels,bias=True),
                nn.SiLU(),
                nn.Conv2d(hidden_channels,1,kernel_size=1,bias=True),
            )
        else:
            raise ValueError(f"{mode} doesn't exist, use 'basic' or 'enhanced'")

        #gate init-zero weights, warm bias
        finalConv=self.gateNet[-1]
        nn.init.zeros_(finalConv.weight)
        nn.init.constant_(finalConv.bias,-1.0)

    def _fusionInput(self,audio_feat,visual_feat,alpha=None):
        if self.mode=="basic" or not self.useInteractions:
            inputs=[audio_feat,visual_feat]
        else:
            inputs=[audio_feat,visual_feat,
                    audio_feat-visual_feat,
                    audio_feat*visual_feat]
        if alpha is not None:
            inputs.append(alpha)
        if self.useEnergySaliency:
            energy=audio_feat.pow(2).mean(dim=1,keepdim=True)
            frameMean=energy.mean(dim=-1,keepdim=True).clamp(min=1e-8)
            logRatio=torch.log(energy.clamp(min=1e-8))-torch.log(frameMean)
            saliency=torch.sigmoid(logRatio.clamp(-4.0,4.0))
            inputs.append(saliency)
        return torch.cat(inputs,dim=1)

    def _logits(self,audio_feat,visual_feat=None,alpha=None):
        if visual_feat is None:
            visual_feat=torch.zeros_like(audio_feat)
        return self.gateNet(self._fusionInput(audio_feat,visual_feat,alpha))

    def forward(self,audio_feat,visual_feat=None,alpha=None):
        return torch.sigmoid(self._logits(audio_feat,visual_feat,alpha))

class FSVG_start_bias(FSVG):
    #learnable frequency bias,speech band gets higher init
    def __init__(self,in_channels=None,n_freq=100,sample_rate=16000,
                 speech_band_hz=(300.0,3400.0),init_in_band=0.8,init_out_band=-0.2,
                 use_speech_bias_init=True,mode="basic",hidden_channels=None,
                 context_kernel=3,use_interactions=True,audio_channels=None,
                 visual_channels=None):
        if in_channels is None:
            in_channels=audio_channels
        super().__init__(in_channels=in_channels,mode=mode,
            hidden_channels=hidden_channels,context_kernel=context_kernel,
            use_interactions=use_interactions)
        maxHz=float(sample_rate)*0.5
        hz=torch.linspace(0.0,maxHz,n_freq)
        loHz=max(0.0,float(speech_band_hz[0]))
        hiHz=min(maxHz,float(speech_band_hz[1]))
        speechMask=(hz>=loHz)&(hz<=hiHz)
        self.register_buffer("speechMask",speechMask.view(1,1,1,n_freq),persistent=False)
        prior=torch.zeros(1,1,1,n_freq)
        if use_speech_bias_init:
            prior[...,speechMask]=float(init_in_band)
            prior[...,~speechMask]=float(init_out_band)
        self.freqPrior=nn.Parameter(prior)

    def _logits(self,audio_feat,visual_feat,alpha=None):
        return super()._logits(audio_feat,visual_feat,alpha)+self.freqPrior

class FSVGLite(nn.Module):
    #single conv+freq prior 130 params
    def __init__(self,in_channels=64,n_freq=100,sample_rate=16000,
                 speech_band_hz=(300.0,4000.0),init_in_band=0.5,init_out_band=-0.3):
        super().__init__()
        self.gateConv=nn.Conv2d(2*in_channels,1,kernel_size=1,bias=True)
        maxHz=float(sample_rate)*0.5
        hz=torch.linspace(0.0,maxHz,n_freq)
        loHz=max(0.0,float(speech_band_hz[0]))
        hiHz=min(maxHz,float(speech_band_hz[1]))
        speechMask=(hz>=loHz)&(hz<=hiHz)
        prior=torch.zeros(1,1,1,n_freq)
        prior[...,speechMask]=float(init_in_band)
        prior[...,~speechMask]=float(init_out_band)
        self.freqPrior=nn.Parameter(prior)

    def forward(self,audio_feat,visual_feat):
        fusion=torch.cat([audio_feat,visual_feat],dim=1)
        return torch.sigmoid(self.gateConv(fusion)+self.freqPrior)
