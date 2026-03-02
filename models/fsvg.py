# Dominic part
# FSVG controls how much visual info gets injected at each frequency bin.
# The idea is lip movements help most with speech frequencies (300Hz-3kHz)
# but don't do much for high-freq noise, so the gate learns to be selective.
import torch
import torch.nn as nn

# FSVG produces a gate [B, 1, T, F] in [0,1] from audio and visual features.
# It's a soft mask that controls which T-F bins get visual info.
# Input is audio_feat [B, C, T, F] and visual_feat [B, C, T, F].
# This works on the T-F plane so use Conv2d, not Conv1d.
# FSVGWithPrior inherits FSVG and adds a learnable frequency prior.
# Think about where to inject the prior so it interacts well with the gating.
# _logits() is split out from forward() so subclasses can override the gating logic.

#Batch size, channels, timeframes, frequency bins.

class FSVG(nn.Module): #outputs mask telling model how much visual info to use per time-frequency bin
    def __init__(self, in_channels): #feature depth of audio and visual maps
        super().__init__()
        hidden_channels = max(8, in_channels // 2)
        
        
        self.gate_net = nn.Sequential(  #defines feed-forward netwrok for gate logits (small nn takes input features and computes outputs in one forward pass)

            nn.Conv2d(2 * in_channels, hidden_channels, kernel_size=1, bias=True), #1x1 mixes audio visual channels cheaply
            #The input at each time, frequency bin has 2C number of values (audio channels + visual channels), 1x1 conv applies learned linear mix across those 2C channels
            #out put becomes [hidden channels] features per bin

            #needed as lets gate compare/combine audio and visual evidence channel by channel
            #reduces dimension from 2C -> hidden before later layers, so computation is cheaper
            nn.SiLU(), #activation function
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True), #input, output, bias
        )

    def _logits(self, audio_feat, visual_feat):
        fusion_in = torch.cat([audio_feat, visual_feat], dim=1) #joins tensors directly
        return self.gate_net(fusion_in)

    def forward(self, audio_feat, visual_feat):
        return torch.sigmoid(self._logits(audio_feat, visual_feat))


class FSVGWithPrior(FSVG): #learnable frequency bias added - adds frequency bias to learn, need to add into optimizer in training.
    def __init__(self, in_channels, n_freq):
        super().__init__(in_channels)
        self.freq_prior = nn.Parameter(torch.zeros(1, 1, 1, n_freq))

    def _logits(self, audio_feat, visual_feat):
        logits = super()._logits(audio_feat, visual_feat)
        return logits + self.freq_prior