#author:Billy
#SEMamba part original code
# LiteAVSEMamba is the Lite AV real time version
#unified implementation: YAML switches control Scout, AudioNeed, VCE type
#use_cross_modal_vce:true/false - CrossModalVCE vs VCEWithTemporalSmoothing
#scout_mode:gate_only/none - Scout lookahead on/off
#use_audio_need:true/false - AudioNeed amplifier on/off
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .mamba_block import TFMambaBlock,CausalTFMambaBlock
from .codec_module import DenseEncoder,MagDecoder,PhaseDecoder
from .vce import CrossModalVCE,VCEWithTemporalSmoothing
from .fsvg import FSVG
from .lite_visual_encoder import create_visual_encoder
from .audio_need import AudioNeedEstimator

#Original SEMamba (audio-only baseline)-No touch!
class SEMamba(nn.Module):
    def __init__(self,cfg):
        super(SEMamba,self).__init__()
        self.cfg=cfg
        self.num_tscblocks=cfg['model_cfg']['num_tfmamba'] if cfg['model_cfg']['num_tfmamba'] is not None else 4
        self.dense_encoder=DenseEncoder(cfg)
        self.TSMamba=nn.ModuleList([TFMambaBlock(cfg) for _ in range(self.num_tscblocks)])
        self.mask_decoder=MagDecoder(cfg)
        self.phase_decoder=PhaseDecoder(cfg)

    def forward(self,noisy_mag,noisy_pha):
        noisy_mag=rearrange(noisy_mag,'b f t -> b t f').unsqueeze(1)
        noisy_pha=rearrange(noisy_pha,'b f t -> b t f').unsqueeze(1)
        x=torch.cat((noisy_mag,noisy_pha),dim=1)
        x=self.dense_encoder(x)
        for block in self.TSMamba:
            x=block(x)
        denoised_mag=rearrange(self.mask_decoder(x)*noisy_mag,'b c t f -> b f t c').squeeze(-1)
        denoised_pha=rearrange(self.phase_decoder(x),'b c t f -> b f t c').squeeze(-1)
        denoised_com=torch.stack(
            (denoised_mag*torch.cos(denoised_pha),denoised_mag*torch.sin(denoised_pha)),
            dim=-1)
        return denoised_mag,denoised_pha,denoised_com

#LiteAVSEMamba(unified AV speech enhancement)
#YAML switches
#use_cross_modal_vce - CrossModalVCE(visual+audio) or VCEWithTemporalSmoothing(visual only)
#scout_mode - gate_only=on,none=off
#use_audio_need - AudioNeed amplifier on/off
class LiteAVSEMamba(nn.Module):
    def __init__(self,cfg):
        super(LiteAVSEMamba,self).__init__()
        self.cfg=cfg
        self.hid=cfg['model_cfg']['hid_feature']
        self.useVisual=cfg.get('visual_cfg',{}).get('use_visual',True)
        liteCfg=cfg.get('lite_cfg',{})
        self.visDropRate=liteCfg.get('visual_dropout_rate',0.0)
        self.currentStep=0
        self.useCrossModalVCE=liteCfg.get('use_cross_modal_vce',True)
        self.useGate=liteCfg.get('use_unified_gate',True)

        #audio backbone
        self.dense_encoder=DenseEncoder(cfg)
        self.mask_decoder=MagDecoder(cfg)
        self.phase_decoder=PhaseDecoder(cfg)

        #mamba blocks
        nBlocks=cfg['model_cfg'].get('num_tfmamba',4)
        self.TSMamba=nn.ModuleList([CausalTFMambaBlock(cfg) for _ in range(nBlocks)])

        #visual branch
        if self.useVisual:
            self.visualEncoder=create_visual_encoder(cfg)

            #VCE has cross modal or visual only functionality(controlled by use_cross_modal_vce)
            if self.useCrossModalVCE:
                self.VCE=CrossModalVCE(
                    visual_dim=512,audio_dim=self.hid,
                    hidden_dim=128,smooth_kernel=5)
            else:
                self.VCE=VCEWithTemporalSmoothing(
                    input_dim=512,hidden_dims=[256,64],
                    smooth_kernel=5)

            #early fusion projections
            nFreq=cfg['stft_cfg']['n_fft']//2+1
            self.audioEarlyProj=nn.Linear(nFreq,self.hid)
            self.visualEarlyProj=nn.Conv1d(512,self.hid,kernel_size=1)
            #FSVG is a freqwise gate, and alpha as unified input
            self.FSVG=FSVG(
                in_channels=self.hid,mode="enhanced",
                context_kernel=3,use_interactions=True,
                alpha_channels=1)
            #visual(1ch for concat with mag+pha)
            self.visualChannelProj=nn.Conv2d(self.hid,1,kernel_size=1)
            #aux visual loss head
            self.visualProj=nn.Sequential(
                nn.Conv1d(512,self.hid,kernel_size=1),
                nn.GroupNorm(1,self.hid))
            self.visualAuxHead=nn.Linear(self.hid,self.hid)
            #scout:VCE looks ahead k frames(off when scout_mode=none)
            self.scoutMode=liteCfg.get('scout_mode','none')
            self.scoutFrames=liteCfg.get('scout_frames',2)

            #AudioNeed amplifier(off when use_audio_need=false)
            #w=r_vis*r_freq*(1+alpha*r_need)
            self.useAudioNeed=liteCfg.get('use_audio_need',False)
            if self.useAudioNeed:
                nFreqRaw=cfg['stft_cfg']['n_fft']//2+1
                self.AudioNeed=AudioNeedEstimator(
                    hidden_dim=16,
                    warmup_steps=liteCfg.get('audio_need_warmup',5000),
                    n_freq=nFreqRaw,
                    sample_rate=cfg['stft_cfg']['sampling_rate'])
                self.AudioNeedAlpha=liteCfg.get('audio_need_alpha',0.75)

    def _visualDropout(self,vis,B):
        #contiguous temporal mask dropout
        if not self.training or self.visDropRate<=0:
            return vis
        Tv=vis.shape[2]
        maskLen=max(2,int(Tv*self.visDropRate))
        if Tv<=maskLen:
            return vis
        mask=torch.ones(B,1,Tv,device=vis.device)
        starts=torch.randint(0,Tv-maskLen,(B,))
        for i in range(B):
            mask[i,:,starts[i]:starts[i]+maskLen]=0.0
        return vis*mask

    def _scout(self,visRaw):
        #lookahead:avgpool over[t,t+1,...,t+k]
        if self.scoutMode=='none' or self.scoutFrames<=0:
            return visRaw
        k=self.scoutFrames
        vPad=F.pad(visRaw,(0,k))
        return F.avg_pool1d(vPad,kernel_size=k+1,stride=1)

    def forward(self,noisy_mag,noisy_pha,video=None,
                return_intermediates=False,visual_degraded=None):
        intermediates={}
        noisy_mag=rearrange(noisy_mag,'b f t -> b t f').unsqueeze(1)
        noisy_pha=rearrange(noisy_pha,'b f t -> b t f').unsqueeze(1)

        #early fusion path is visual enters before DenseEncoder
        if self.useVisual and video is not None:
            B=noisy_mag.shape[0]
            T=noisy_mag.shape[2]
            Freq=noisy_mag.shape[3]

            #visual encoder
            visRaw=self.visualEncoder(video,T)
            visRaw=self._visualDropout(visRaw,B)

            #scout means lookahead for VCE(no op when scout_mode=none)
            scoutVis=self._scout(visRaw)

            #project visual for fusion
            visForFusion=scoutVis if self.scoutMode=='full' else visRaw
            visProj=self.visualEarlyProj(visForFusion)
            vis2d=visProj.unsqueeze(-1).expand(-1,-1,-1,Freq)

            if self.useGate:
                #VCE temporal confidence alpha[B,T,1]
                visForVCE=scoutVis.permute(0,2,1)
                audioProj=self.audioEarlyProj(noisy_mag.squeeze(1))
                if self.useCrossModalVCE:
                    alpha=self.VCE(visForVCE,audioProj)
                else:
                    alpha=self.VCE(visForVCE)

                if return_intermediates:
                    intermediates['alpha']=alpha.detach()
                    intermediates['visual_raw']=visRaw.detach()

                #gate supervision
                if self.training and visual_degraded is not None:
                    gateTarget=torch.where(
                        visual_degraded.bool(),
                        torch.tensor(0.15,device=alpha.device),
                        torch.tensor(0.85,device=alpha.device))
                    alphaMean=alpha.mean(dim=1).squeeze(-1)
                    intermediates['gate_loss']=F.binary_cross_entropy(alphaMean,gateTarget)

                #freq wise gate [B,1,T,F]
                audio2d=audioProj.permute(0,2,1).unsqueeze(-1).expand(-1,-1,-1,Freq)
                alphaFeat=alpha.permute(0,2,1).unsqueeze(-1).expand(-1,-1,-1,Freq)
                freqGate=self.FSVG(audio2d,vis2d,alpha=alphaFeat)

                if return_intermediates:
                    intermediates['freq_gate']=freqGate.detach()

                #gating:w=alpha*freqGate(*needAmp)*visual
                alphaBc=alpha.permute(0,2,1).unsqueeze(-1)
                if self.useAudioNeed and hasattr(self,'AudioNeed'):
                    step=getattr(self,'currentStep',0)
                    rNeed=self.AudioNeed(noisy_mag,step=step)
                    needAmp=1.0+self.AudioNeedAlpha*rNeed
                    needAmpBc=needAmp.permute(0,2,1).unsqueeze(-1)
                    visGated=alphaBc*freqGate*needAmpBc*vis2d
                    if return_intermediates:
                        intermediates['r_need']=rNeed.detach()
                        intermediates['need_amplifier']=needAmp.detach()
                else:
                    visGated=alphaBc*freqGate*vis2d
            else:
                #direct fusion: no VCE, no FSVG, just project visual
                visGated=vis2d

            #project to 1ch and concat
            visCh=self.visualChannelProj(visGated)
            x=torch.cat((noisy_mag,noisy_pha,visCh),dim=1)
            x=self.dense_encoder(x)
            B,C,T,Fenc=x.shape

            #aux visual loss
            audioTarget=x.mean(dim=-1).detach()
            visAuxFeat=self.visualProj(visRaw).permute(0,2,1)
            intermediates['aux_loss']=F.mse_loss(
                self.visualAuxHead(visAuxFeat),audioTarget.permute(0,2,1))

        else:
            #audioonly fallback-pad zero channel for input_channel=3
            zeroCh=torch.zeros_like(noisy_mag)
            x=torch.cat((noisy_mag,noisy_pha,zeroCh),dim=1)
            x=self.dense_encoder(x)
            B,C,T,Fenc=x.shape
        #mamba blocks
        for block in self.TSMamba:
            x=block(x)
        #decode
        denoised_mag=rearrange(self.mask_decoder(x)*noisy_mag,'b c t f -> b f t c').squeeze(-1)
        denoised_pha=rearrange(self.phase_decoder(x),'b c t f -> b f t c').squeeze(-1)
        denoised_com=torch.stack(
            (denoised_mag*torch.cos(denoised_pha),denoised_mag*torch.sin(denoised_pha)),
            dim=-1)

        if return_intermediates:
            return denoised_mag,denoised_pha,denoised_com,intermediates
        return denoised_mag,denoised_pha,denoised_com
