# Dominic part
# FSVG outputs gate determining how much visual info gets injected at each frequency bin.
# The idea is lip movements help most with speech frequencies (300Hz-3kHz)
import torch
import torch.nn as nn

#GENERAL GIST...
#lip reading done by encoder provides visual tensors that are learned to correspond to 

#Batch size, channels, timeframes, frequency bins.

#BASED ON GMU.

#We don’t want a rough gate (sharp jumps between neighboring bins). 
#So we let each gate value use neighbouring information, which makes the gate smoother and more stable.
#why we use context kernel for enhanced.
class FSVG(nn.Module): #outputs mask telling model how much visual info to use per time-frequency bin
    def __init__(
        self,
        in_channels,
        mode="basic",
        hidden_channels=None,
        context_kernel=3,
        use_interactions=True,
    ): #feature depth of audio and visual maps
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(8, in_channels // 2) #sets num of hidden channels for model

        if context_kernel % 2 == 0: #must be odd so output stays the same size after convolution, padding is symetric //need a centre of convolution window.
            raise ValueError("context_kernel must be odd so spatial size is preserved")

        self.mode = str(mode).lower()
        self.use_interactions = bool(use_interactions)

        if self.mode == "basic": #setting up the type of model
            self.gate_net = nn.Sequential( #Classic CNN
                nn.Conv2d(2 * in_channels, hidden_channels, kernel_size=1, bias=True),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
            )
        elif self.mode == "enhanced": #setting up the type of model
            if self.use_interactions: #if we want to use cross-modal interaction features
                fusion_channels = 4 * in_channels  #audio, visual, difference, product
            else:
                fusion_channels = 2 * in_channels  #audio and visual only
            self.gate_net = nn.Sequential( #2d CNN 
                nn.Conv2d(fusion_channels, hidden_channels, kernel_size=1, bias=True),
                nn.SiLU(),
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=context_kernel,
                    padding=context_kernel // 2,
                    groups=hidden_channels,
                    bias=True,
                ),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
            )
        else:
            raise ValueError(f"{mode}. Doesn't exist Use 'basic' or 'enhanced'.")

    def _fusion_input(self, audio_feat, visual_feat):
        if self.mode == "basic" or not self.use_interactions:
            #basic mode just concat audio and visual side by side. [B, 2C, T, F]
            return torch.cat([audio_feat, visual_feat], dim=1)
        #enhanced: add explicit interaction terms so the gate can reason
        #whether audio and visual agree at each T-F bin.
        return torch.cat(
            [
                audio_feat,
                visual_feat,
                audio_feat - visual_feat,   # disagreement signal
                audio_feat * visual_feat,   # co-activation signal
            ],
            dim=1,
        )

    def _logits(self, audio_feat, visual_feat=None):
        if visual_feat is None:
            visual_feat = torch.zeros_like(audio_feat)
        fusion_in = self._fusion_input(audio_feat, visual_feat)
        return self.gate_net(fusion_in)

    def forward(self, audio_feat, visual_feat=None):
        #sigmoid squashes logits to [0, 1], values for gate have to be [0,1]
        return torch.sigmoid(self._logits(audio_feat, visual_feat))


class FSVGEnhanced(FSVG):
    def __init__(self, in_channels, hidden_channels=None, context_kernel=3, use_interactions=True):
        super().__init__(
            in_channels=in_channels,
            mode="enhanced",
            hidden_channels=hidden_channels,
            context_kernel=context_kernel,
            use_interactions=use_interactions,
        )


class FSVG_start_bias(FSVG): #learnable frequency bias added - adds frequency bias to learn, need to add into optimizer in training.
    def __init__(
        self,
        in_channels,
        n_freq,
        sample_rate=16000,
        speech_band_hz=(300.0, 3400.0),
        init_in_band=0.8,
        init_out_band=-0.2,
        use_speech_bias_init=True,
        mode="basic",
        hidden_channels=None,
        context_kernel=3,
        use_interactions=True,
    ):
        super().__init__(
            in_channels=in_channels,
            mode=mode,
            hidden_channels=hidden_channels,
            context_kernel=context_kernel,
            use_interactions=use_interactions,
        )
        #build a frequency axis from 0 Hz to Nyquist with n_freq points,
        #matching the F dimension of the encoded feature maps.
        max_hz = float(sample_rate) * 0.5   #shannon-nyquist
        hz = torch.linspace(0.0, max_hz, n_freq)

        #clamp the speech band to valid Hz range in case of unusual config values.
        low_hz, high_hz = speech_band_hz
        low_hz = max(0.0, float(low_hz))
        high_hz = min(max_hz, float(high_hz))

        #boolean mask: True at frequency bins inside the speech band.
        speech_band_mask = (hz >= low_hz) & (hz <= high_hz)
        #register as a buffer so it moves to the right device automatically,
        #but persistent=False means it won't be saved in checkpoints (it's
        #always re-derived from the constructor args).
        
        self.register_buffer("speech_band_mask", speech_band_mask.view(1, 1, 1, n_freq), persistent=False)

        #Each frequency has its own bias, the model starts with preference skewed to where the speech band is
        prior = torch.zeros(1, 1, 1, n_freq)
        if use_speech_bias_init:
            prior[..., speech_band_mask] = float(init_in_band)    #changing prior tensor values (100 freq bins), if freq bin falls within speech band mask set to bias +0.8
            prior[..., ~speech_band_mask] = float(init_out_band)  #falls outside set to negative bias weight. -0.2
        self.freq_prior = nn.Parameter(prior)  #this needs to be a learnable paramter by the optimizer, so wrap it inside parameter

    def _logits(self, audio_feat, visual_feat):
        logits = super()._logits(audio_feat, visual_feat)
        return logits + self.freq_prior


# Backward-compatible name used by tests/demo code.
FSVGWithPrior = FSVG_start_bias


class FSVGEnhancedWithPrior(FSVG_start_bias):

    def __init__(
        self,
        in_channels,
        n_freq,
        sample_rate=16000,
        speech_band_hz=(300.0, 3400.0),
        init_in_band=0.8,
        init_out_band=-0.2,
        use_speech_bias_init=True,
        hidden_channels=None,
        context_kernel=3,
        use_interactions=True,
    ):
        super().__init__(
            in_channels=in_channels,
            n_freq=n_freq,
            sample_rate=sample_rate,
            speech_band_hz=speech_band_hz,
            init_in_band=init_in_band,
            init_out_band=init_out_band,
            use_speech_bias_init=use_speech_bias_init,
            mode="enhanced",
            hidden_channels=hidden_channels,
            context_kernel=context_kernel,
            use_interactions=use_interactions,
        )
