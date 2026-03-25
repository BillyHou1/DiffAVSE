# Zhenning part
# ShuffleNetV2 visual encoder (default)
# Also keeps LiteVisualEncoderA (MobileNetV3-Small) and LiteVisualEncoderB (custom 3D CNN) as alternatives.
#
# ShuffleNetV2: [B, 3, T_v, 96, 96] -> [B, 512, T_audio]
# - RGB input converted to grayscale inside model
# - Pretrained LRW lipreading weights (grayscale) for frontend3D + trunk
# - frontend3D & trunk are frozen by default; LiteTCN is trainable
# - LiteTCN: bottleneck TCN with residual, receptive field 29 frames

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ShuffleNetV2 building blocks
# ============================================================

def _channelShuffle(x, groups):
    B, C, H, W = x.shape
    x = x.view(B, groups, C // groups, H, W)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(B, -1, H, W)


class _InvertedResidual(nn.Module):
    """ShuffleNetV2 building block, matching mpc001."""

    def __init__(self, inp, oup, stride, benchmodel):
        super().__init__()
        self.benchmodel = benchmodel
        oupInc = oup // 2
        if benchmodel == 1:
            self.banch2 = nn.Sequential(
                nn.Conv2d(oupInc, oupInc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oupInc), nn.ReLU(inplace=True),
                nn.Conv2d(oupInc, oupInc, 3, stride, 1, groups=oupInc, bias=False),
                nn.BatchNorm2d(oupInc),
                nn.Conv2d(oupInc, oupInc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oupInc), nn.ReLU(inplace=True)
            )
        else:
            self.banch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oupInc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oupInc), nn.ReLU(inplace=True)
            )
            self.banch2 = nn.Sequential(
                nn.Conv2d(inp, oupInc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oupInc), nn.ReLU(inplace=True),
                nn.Conv2d(oupInc, oupInc, 3, stride, 1, groups=oupInc, bias=False),
                nn.BatchNorm2d(oupInc),
                nn.Conv2d(oupInc, oupInc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oupInc), nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.benchmodel == 1:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = torch.cat((x1, self.banch2(x2)), 1)
        else:
            out = torch.cat((self.banch1(x), self.banch2(x)), 1)
        return _channelShuffle(out, 2)


# ============================================================
# LiteTCN: bottleneck TCN for temporal modeling
# ============================================================

class LiteTCN(nn.Module):
    """Bottleneck TCN: compress -> 3x dilated depthwise -> expand.
    Receptive field 29 frames.
    """

    def __init__(self, inCh=512, bottleneck=64, dilations=(1, 2, 4)):
        super().__init__()
        self.compress = nn.Conv1d(inCh, bottleneck, 1)
        self.tcnLayers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(bottleneck, bottleneck, 5, dilation=d,
                          padding=2 * d, groups=bottleneck),
                nn.GroupNorm(4, bottleneck),
                nn.SiLU(),
            ) for d in dilations
        ])
        self.expand = nn.Conv1d(bottleneck, inCh, 1)
        # Zero-init the residual branch so it starts as identity
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, x):
        h = self.compress(x)
        for layer in self.tcnLayers:
            h = h + layer(h)
        return x + self.expand(h)


# ============================================================
# ShuffleNetVisualEncoder (default)
# ============================================================

class ShuffleNetVisualEncoder(nn.Module):
    """
    ShuffleNetV2 + LiteTCN visual encoder.
    Input:  [B, 3, T_v, 96, 96]  (RGB, Full-face ROI, center-crop + resize to 96x96)
    Output: [B, 512, T_audio]

    - RGB->grayscale inside model
    - frontend3D + trunk: loaded from LRW lipreading pretrained weights, frozen by default
    - LiteTCN: trainable (residual bottleneck TCN)
    """

    def __init__(self, cfg):
        super().__init__()
        visCfg = cfg.get('visual_cfg', {})

        # 3D frontend: Conv3d(1, 24, ...) — grayscale input
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=(5, 7, 7), stride=(1, 2, 2),
                      padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(24),
            nn.PReLU(24),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        # ShuffleNetV2 trunk
        stageRepeats = [4, 8, 4]
        stageCh = [24, 116, 232, 464]
        features = []
        inCh = 24
        for idx in range(3):
            outCh = stageCh[idx + 1]
            for i in range(stageRepeats[idx]):
                if i == 0:
                    features.append(_InvertedResidual(inCh, outCh, 2, 2))
                else:
                    features.append(_InvertedResidual(inCh, outCh, 1, 1))
                inCh = outCh
        convLast = nn.Sequential(
            nn.Conv2d(464, 1024, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        globalPool = nn.Sequential(nn.AvgPool2d(3))
        self.trunk = nn.Sequential(nn.Sequential(*features), convLast, globalPool)

        # Channel projection + LiteTCN
        self.channelProj = nn.Conv1d(1024, 512, kernel_size=1)
        tcnBottleneck = visCfg.get('tcn_bottleneck', 64)
        self.temporalHead = LiteTCN(inCh=512, bottleneck=tcnBottleneck, dilations=(1, 2, 4))

        # Optional temporal shift for alignment
        self.temporalShift = visCfg.get('visual_temporal_shift', 0)

        # Freeze frontend3D and trunk by default
        if visCfg.get('freeze_visual_encoder', True):
            for p in self.frontend3D.parameters():
                p.requires_grad = False
            for p in self.trunk.parameters():
                p.requires_grad = False

        # Load LRW lipreading pretrained weights if path provided
        pretrainedPath = visCfg.get('lipreading_weights', None)
        if pretrainedPath:
            self._loadWeights(pretrainedPath)

    def _loadWeights(self, path):
        """Load pretrained LRW lipreading weights into frontend3D + trunk."""
        if not os.path.exists(path):
            print(f"WARNING: weights not found at {path}")
            return
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        filtered = {k: v for k, v in state.items()
                    if k.startswith('frontend3D.') or k.startswith('trunk.')}
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(f"ShuffleNetV2: loaded {len(filtered) - len(unexpected)}/{len(filtered)} params")

    def _preprocess(self, video):
        """RGB -> grayscale: [B, 3, T, H, W] -> [B, 1, T, H, W]"""
        return 0.299 * video[:, 0:1] + 0.587 * video[:, 1:2] + 0.114 * video[:, 2:3]

    def train(self, mode=True):
        """Keep frozen modules in eval mode during training."""
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

    def forward(self, video, T_audio):
        """
        Args:
            video:   [B, 3, T_v, 96, 96]
            T_audio: int, target audio STFT time steps
        Returns:
            [B, 512, T_audio]
        """
        B, C, Tv, H, W = video.shape

        # RGB -> grayscale
        gray = self._preprocess(video)  # [B, 1, Tv, 96, 96]

        # 3D frontend: use no_grad only when frozen
        if all(not p.requires_grad for p in self.frontend3D.parameters()):
            with torch.no_grad():
                x = self.frontend3D(gray)
        else:
            x = self.frontend3D(gray)
        # Reshape for 2D trunk: [B, C2, T2, H2, W2] -> [B*T2, C2, H2, W2]
        _, C2, T2, H2, W2 = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T2, C2, H2, W2)

        # ShuffleNetV2 trunk: use no_grad only when frozen
        if all(not p.requires_grad for p in self.trunk.parameters()):
            with torch.no_grad():
                x = self.trunk(x)
        else:
            x = self.trunk(x)

        # Reshape to temporal: [B*T2, 1024, 1, 1] -> [B, T2, 1024] -> [B, 1024, T2]
        x = x.view(B * T2, -1)
        features = x.view(B, T2, -1).permute(0, 2, 1)

        # Channel projection: [B, 1024, T2] -> [B, 512, T2]
        features = self.channelProj(features)

        # Optional temporal shift for alignment
        if self.temporalShift > 0 and features.shape[2] > self.temporalShift:
            features = F.pad(features[:, :, :-self.temporalShift],
                             (self.temporalShift, 0))

        # LiteTCN temporal modeling (trainable)
        features = self.temporalHead(features)  # [B, 512, T2]

        # Interpolate to T_audio: [B, 512, T2] -> [B, 512, T_audio]
        return F.interpolate(features, size=T_audio, mode='linear', align_corners=False)


# ============================================================
# LiteVisualEncoderA (MobileNetV3-Small, legacy)
# ============================================================

class LiteVisualEncoderA(nn.Module):
    """
    Legacy visual encoder A: pretrained MobileNetV3-Small + temporal Conv1d.
    Input:  [B, 3, T_v, 96, 96]
    Output: [B, 512, T_audio]
    """

    def __init__(self, cfg):
        super().__init__()
        from torchvision import models

        backbone = models.mobilenet_v3_small(pretrained=True)
        self.features = backbone.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(576, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        if cfg.get('visual_cfg', {}).get('freeze_visual_encoder', True):
            self.freeze_visual_encoder()

    def freeze_visual_encoder(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, video, T_audio):
        B, C, T_v, H, W = video.shape

        frames = video.permute(0, 2, 1, 3, 4).contiguous()
        frames = frames.view(-1, C, H, W)

        spatial_features = self.features(frames)
        spatial_features = self.spatial_pool(spatial_features)
        spatial_features = spatial_features.view(B * T_v, -1)
        spatial_features = spatial_features.view(B, T_v, -1).permute(0, 2, 1)

        temporal_features = self.temporal_conv(spatial_features)
        output = F.interpolate(temporal_features, size=T_audio, mode='linear', align_corners=False)

        return output


# ============================================================
# LiteVisualEncoderB (Custom 3D CNN, legacy)
# ============================================================

class LiteVisualEncoderB(nn.Module):
    """
    Legacy visual encoder B: custom lightweight 3D CNN, trained from scratch.
    Input:  [B, 3, T_v, 96, 96]
    Output: [B, 512, T_audio]
    Params: < 1M
    """

    def __init__(self, cfg):
        super().__init__()

        self.conv3d_backbone = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        self.temporal_projection = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=1)
        )

    def forward(self, video, T_audio):
        B, C, T_v, H, W = video.shape

        features_3d = self.conv3d_backbone(video)
        features = self.spatial_pool(features_3d)
        features = features.squeeze(-1).squeeze(-1)

        temporal_features = self.temporal_projection(features)
        output = F.interpolate(temporal_features, size=T_audio, mode='linear', align_corners=False)

        return output


# ============================================================
# Factory
# ============================================================

def create_visual_encoder(cfg):
    """
    Create visual encoder based on config.

    Args:
        cfg: config dict, visual_cfg.encoder_type selects the encoder
             - 'shuffle'  (default): ShuffleNetV2 + LiteTCN
             - 'A': MobileNetV3-Small (legacy)
             - 'B': Custom 3D CNN (legacy)
    Returns:
        nn.Module instance
    """
    encoder_type = cfg.get('visual_cfg', {}).get('encoder_type', 'shuffle')

    if encoder_type == 'shuffle':
        return ShuffleNetVisualEncoder(cfg)
    elif encoder_type == 'A':
        return LiteVisualEncoderA(cfg)
    elif encoder_type == 'B':
        return LiteVisualEncoderB(cfg)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                         f"Choose from 'shuffle', 'A', 'B'.")


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    cfg = {
        'visual_cfg': {
            'encoder_type': 'shuffle',
            'freeze_visual_encoder': True,
        }
    }

    batch_size = 2
    T_video = 75   # 3s @ 25fps
    T_audio = 480  # 3s @ 160fps

    video_input = torch.randn(batch_size, 3, T_video, 96, 96)

    # Test ShuffleNetV2 (default)
    encoder = ShuffleNetVisualEncoder(cfg)
    output = encoder(video_input, T_audio)
    print(f"ShuffleNetV2 input:  {video_input.shape}")
    print(f"ShuffleNetV2 output: {output.shape}")

    total = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}  (LiteTCN)")
    print(f"Frozen params:    {frozen:,}  (frontend3D + trunk)")

    print("\nTrainable:")
    for n, p in encoder.named_parameters():
        if p.requires_grad:
            print(f"  {n}")
