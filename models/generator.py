# Billy
# SEMamba below is the original audio-only model, already done.
# LiteAVSEMamba is added at the bottom of this file.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .mamba_block import TFMambaBlock, CausalTFMambaBlock
from .codec_module import DenseEncoder, MagDecoder, PhaseDecoder
from .codec_module import LiteDenseEncoder, LiteMagDecoder, LitePhaseDecoder
from .vce import VCE   # swap with VCEWithTemporalSmoothing if you want smoothing
from .fsvg import FSVG  # swap with FSVGWithPrior if you want the freq prior
from .lite_visual_encoder import LiteVisualEncoderA, LiteVisualEncoderB

class SEMamba(nn.Module):
    """
    SEMamba model for speech enhancement using Mamba blocks.

    This model uses a dense encoder, multiple Mamba blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """
    def __init__(self, cfg):
        """
        Initialize the SEMamba model.

        Args:
        - cfg: Configuration object containing model parameters.
        """
        super(SEMamba, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = cfg['model_cfg']['num_tfmamba'] if cfg['model_cfg']['num_tfmamba'] is not None else 4  # default tfmamba: 4

        # Initialize dense encoder
        self.dense_encoder = DenseEncoder(cfg)

        # Initialize Mamba blocks
        self.TSMamba = nn.ModuleList([TFMambaBlock(cfg) for _ in range(self.num_tscblocks)])

        # Initialize decoders
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

    def forward(self, noisy_mag, noisy_pha):
        """
        Forward pass for the SEMamba model.

        Args:
        - noisy_mag (torch.Tensor): Noisy magnitude input tensor [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase input tensor [B, F, T].

        Returns:
        - denoised_mag (torch.Tensor): Denoised magnitude tensor [B, F, T].
        - denoised_pha (torch.Tensor): Denoised phase tensor [B, F, T].
        - denoised_com (torch.Tensor): Denoised complex tensor [B, F, T, 2].
        """
        # Reshape inputs
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]

        # Concatenate magnitude and phase inputs
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]

        # Encode input
        x = self.dense_encoder(x)

        # Apply Mamba blocks
        for block in self.TSMamba:
            x = block(x)

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(x) * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(x), 'b c t f -> b f t c').squeeze(-1)

        # Combine denoised magnitude and phase into a complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)),
            dim=-1
        )

        return denoised_mag, denoised_pha, denoised_com


# TODO LiteAVSEMamba
# AV version of SEMamba. Fusion happens after DenseEncoder in feature space,
# so input_channel stays 2 (mag+pha), same as SEMamba.
# Visual gets modulated by alpha from VCE and gate from FSVG,
# then added to audio features as a residual.
# Uses CausalTFMambaBlock instead of TFMambaBlock.
# When video=None just skip the visual branch entirely.
# cfg['visual_cfg']['use_visual'] turns the visual branch on/off
# cfg['lite_cfg']['visual_encoder_type'] picks EncoderA or EncoderB
# cfg['lite_cfg']['n_freq_enc'] is the freq dim after DenseEncoder
class LiteAVSEMamba(nn.Module):
    def __init__(self, cfg):
        super(LiteAVSEMamba, self).__init__()
        self.cfg = cfg
        # TODO audio backbone (refer to SEMamba above for the pattern)
        # TODO visual branch: visual encoder, VCE, FSVG, and projection layers
        raise NotImplementedError
    def forward(self, noisy_mag, noisy_pha, video=None):
        """
        Args:
            noisy_mag: [B, F, T]
            noisy_pha: [B, F, T]
            video:     [B, 3, Tv, 96, 96] or None for audio-only fallback
        Returns:
            denoised_mag, denoised_pha, denoised_com  (same shapes as SEMamba)

        Flow: encode audio → fuse visual if available → CausalTFMamba → decode.
        Handle the case where video is None (audio-only fallback).
        """
        raise NotImplementedError
