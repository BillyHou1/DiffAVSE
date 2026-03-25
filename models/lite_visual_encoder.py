#Reference:https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
#Zhenning part
#ShuffleNetV2 visual encoder
#[B,3,Tv,96,96]->[B,512,T_audio]
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
def _channelShuffle(x,groups):
    B,C,H,W=x.shape
    x=x.view(B,groups,C//groups,H,W)
    x=torch.transpose(x,1,2).contiguous()
    return x.view(B,-1,H,W)

class _InvertedResidual(nn.Module):
    #ShuffleNetV2 building block, matching mpc001
    def __init__(self,inp,oup,stride,benchmodel):
        super().__init__()
        self.benchmodel=benchmodel
        oupInc=oup//2
        if benchmodel==1:
            self.banch2=nn.Sequential(
                nn.Conv2d(oupInc,oupInc,1,1,0,bias=False),
                nn.BatchNorm2d(oupInc),nn.ReLU(inplace=True),
                nn.Conv2d(oupInc,oupInc,3,stride,1,groups=oupInc,bias=False),
                nn.BatchNorm2d(oupInc),
                nn.Conv2d(oupInc,oupInc,1,1,0,bias=False),
                nn.BatchNorm2d(oupInc),nn.ReLU(inplace=True))
        else:
            self.banch1=nn.Sequential(
                nn.Conv2d(inp,inp,3,stride,1,groups=inp,bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp,oupInc,1,1,0,bias=False),
                nn.BatchNorm2d(oupInc),nn.ReLU(inplace=True))
            self.banch2=nn.Sequential(
                nn.Conv2d(inp,oupInc,1,1,0,bias=False),
                nn.BatchNorm2d(oupInc),nn.ReLU(inplace=True),
                nn.Conv2d(oupInc,oupInc,3,stride,1,groups=oupInc,bias=False),
                nn.BatchNorm2d(oupInc),
                nn.Conv2d(oupInc,oupInc,1,1,0,bias=False),
                nn.BatchNorm2d(oupInc),nn.ReLU(inplace=True))

    def forward(self,x):
        if self.benchmodel==1:
            x1=x[:,:(x.shape[1]//2),:,:]
            x2=x[:,(x.shape[1]//2):,:,:]
            out=torch.cat((x1,self.banch2(x2)),1)
        else:
            out=torch.cat((self.banch1(x),self.banch2(x)),1)
        return _channelShuffle(out,2)

class LiteTCN(nn.Module):
    #bottleneck TCN(compress->3x dilated depthwise->expand)
    #receptive field 29 frames
    def __init__(self,inCh=512,bottleneck=64,dilations=(1,2,4)):
        super().__init__()
        self.compress=nn.Conv1d(inCh,bottleneck,1)
        self.tcnLayers=nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(bottleneck,bottleneck,5,dilation=d,
                          padding=2*d,groups=bottleneck),
                nn.GroupNorm(4,bottleneck),
                nn.SiLU(),
            ) for d in dilations
        ])
        self.expand=nn.Conv1d(bottleneck,inCh,1)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self,x):
        h=self.compress(x)
        for layer in self.tcnLayers:
            h=h+layer(h)
        return x+self.expand(h)

class ShuffleNetVisualEncoder(nn.Module):
    #ShuffleNetV2+LiteTCN
    def __init__(self,cfg):
        super().__init__()
        visCfg=cfg.get('visual_cfg',{})
        self.frontend3D=nn.Sequential(
            nn.Conv3d(1,24,kernel_size=(5,7,7),stride=(1,2,2),
                      padding=(2,3,3),bias=False),
            nn.BatchNorm3d(24),
            nn.PReLU(24),
            nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1)))
        #ShuffleNetV2 trunk
        stageRepeats=[4,8,4]
        stageCh=[24,116,232,464]
        features=[]
        inCh=24
        for idx in range(3):
            outCh=stageCh[idx+1]
            for i in range(stageRepeats[idx]):
                if i==0:
                    features.append(_InvertedResidual(inCh,outCh,2,2))
                else:
                    features.append(_InvertedResidual(inCh,outCh,1,1))
                inCh=outCh
        convLast=nn.Sequential(
            nn.Conv2d(464,1024,1,1,0,bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True))
        globalPool=nn.Sequential(nn.AvgPool2d(3))
        self.trunk=nn.Sequential(nn.Sequential(*features),convLast,globalPool)

        self.channelProj=nn.Conv1d(1024,512,kernel_size=1)
        tcnBottleneck=visCfg.get('tcn_bottleneck',64)
        self.temporalHead=LiteTCN(inCh=512,bottleneck=tcnBottleneck,dilations=(1,2,4))
        self.temporalShift=visCfg.get('visual_temporal_shift',0)

        if visCfg.get('freeze_visual_encoder',True):
            for p in self.frontend3D.parameters():
                p.requires_grad=False
            for p in self.trunk.parameters():
                p.requires_grad=False

        pretrainedPath=visCfg.get('lipreading_weights',None)
        if pretrainedPath:
            self._loadWeights(pretrainedPath)

    def _loadWeights(self,path):
        if not os.path.exists(path):
            return
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        state=ckpt.get('model_state_dict',ckpt)
        filtered={k:v for k,v in state.items()
                  if k.startswith('frontend3D.') or k.startswith('trunk.')}
        self.load_state_dict(filtered,strict=False)

    def _preprocess(self,video):
        #RGB->grayscale
        return 0.299*video[:,0:1]+0.587*video[:,1:2]+0.114*video[:,2:3]

    def train(self,mode=True):
        super().train(mode)
        for p in self.frontend3D.parameters():
            if not p.requires_grad:
                self.frontend3D.eval()
                break
        for p in self.trunk.parameters():
            if not p.requires_grad:
                self.trunk.eval()
                break
        return self

    def forward(self,video,T_audio):
        B,C,Tv,H,W=video.shape
        gray=self._preprocess(video)
        with torch.no_grad():
            x=self.frontend3D(gray)
        _,C2,T2,H2,W2=x.shape
        x=x.permute(0,2,1,3,4).contiguous().view(B*T2,C2,H2,W2)
        with torch.no_grad():
            x=self.trunk(x)
        x=x.view(B*T2,-1)
        features=x.view(B,T2,-1).permute(0,2,1)
        features=self.channelProj(features)
        if self.temporalShift>0 and features.shape[2]>self.temporalShift:
            features=F.pad(features[:,:,:-self.temporalShift],
                           (self.temporalShift,0))
        features=self.temporalHead(features)
        return F.interpolate(features,size=T_audio,mode="linear",align_corners=False)

def create_visual_encoder(cfg):
    return ShuffleNetVisualEncoder(cfg)
