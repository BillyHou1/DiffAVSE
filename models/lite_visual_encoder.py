# Zhenning
# Extracts visual features from video frames (lip region) for the audio model.
# Two versions: EncoderA uses a pretrained 2D backbone for transfer learning,
# EncoderB is a custom 3D CNN that trains from scratch.
# Both take video [B, 3, Tv, 96, 96] and output:
# visual_raw [B, 512, T_audio] — generator uses this for VCE and for fusion
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# EncoderA needs: from torchvision import models

# TODO LiteVisualEncoderA and LiteVisualEncoderB
# Both take video [B, 3, Tv, 96, 96] and output [B, 512, T_audio].
# Output is always 512 channels. Use F.interpolate to match the audio frame count.
# EncoderA uses a pretrained 2D backbone to extract per-frame features,
# then some temporal modelling for lip dynamics, then project to 512.
# Most lightweight backbones don't output 512 so you'll need a projection.
# cfg['visual_cfg']['freeze_visual_encoder'] controls whether to freeze the backbone.
# EncoderB is a custom 3D CNN, trains from scratch.
# Downsample spatial dims, pool out H/W, project to 512.
# Try to keep it under 1M params.


class LiteVisualEncoderA(nn.Module):
    """
    Lightweight Visual Encoder A:
    Uses a pretrained MobileNetV3-Small backbone to extract per-frame 2D spatial features.

    Input:  [B, 3, T_v, 96, 96]
    Output:
      - visual_raw: [B, 512, T_audio] for VCE / fusion
    """
    def __init__(self, cfg):
        super().__init__()
        # TODO pretrained backbone, pool spatial, temporal conv, project to 512
        # freeze backbone if cfg says so
        # Load only the backbone "features" module (do not keep the full backbone).
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
       
        # Spatial pooling
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        # Temporal modeling (trainable)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(576, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        # Freeze/unfreeze backbone according to cfg
        if cfg.get('visual_cfg', {}).get('freeze_visual_encoder', True):
            self.freeze_visual_encoder()
        else:
            self.activate_visual_encoder()

    def freeze_visual_encoder(self):
        """Freeze all parameters in the MobileNetV3-Small features backbone."""
        for param in self.features.parameters():
            param.requires_grad = False
    def activate_visual_encoder(self):
        """Unfreeze all parameters in the MobileNetV3-Small features backbone."""
        for param in self.features.parameters():
            param.requires_grad = True
    
     def train(self, mode: bool = True):
         """
         Override train()/eval() switch:
         If the features backbone is frozen, force features.eval() to prevent BatchNorm 
         running stats drifting during training.
         """
         super().train(mode)

         # If features are frozen (requires_grad=False), keep them in eval mode
         # so BN uses running stats instead of updating with batch stats.
         if hasattr(self, "features"):
             for param in self.features.parameters():
                 if not param.requires_grad:
                     self.features.eval()
                     break

         return self
        
     def forward(self, video, T_audio):
        """
        Args:
            video:   [B, 3, Tv, 96, 96]
            T_audio: int, target temporal length (= number of STFT frames)
        Returns:
         [B, 512, T_audio]
        """
          B, C, T_v, H, W = video.shape
        
        # Step 1: per-frame reshape
        # [B, 3, T_v, 96, 96] -> [B*T_v, 3, 96, 96]
        frames = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, T_v, 3, 96, 96]
        frames = frames.view(-1, C, H, W)  # [B*T_v, 3, 96, 96]
        
        # Step 2: 2D spatial feature extraction (MobileNetV3-Small features)
        # [B*T_v, 3, 96, 96] -> [B*T_v, 576, h, w]
        spatial_features = self.features(frames)  # MobileNetV3-Small outputs 576 channels
        
        # Global average pool over H/W
        # [B*T_v, 576, h, w] -> [B*T_v, 576]
        spatial_features = self.spatial_pool(spatial_features)
        spatial_features = spatial_features.view(B * T_v, -1)  # [B*T_v, 576]
        
        # Reshape back to a temporal sequence
        # [B*T_v, 576] → [B, 576, T_v]
        spatial_features = spatial_features.view(B, T_v, -1).permute(0, 2, 1)  # [B, 576, T_v]
        
        # Step 3: temporal modeling (trainable)
        # [B, 576, T_v] -> [B, 512, T_v]
        temporal_features = self.temporal_conv(spatial_features)
        
        # Step 4: align to audio temporal length
        # F.interpolate: [B, 512, T_v] → [B, 512, T_audio]
        visual_raw = F.interpolate(temporal_features, size=T_audio, mode='linear', align_corners=False)
        
        return visual_raw
        

class LiteVisualEncoderB(nn.Module):
    """
    Lightweight Visual Encoder B:
    A custom 3D CNN trained from scratch.

    Input:  [B, 3, T_v, 96, 96]
    Output: [B, 512, T_audio] for VCE / fusion

    Designed to keep parameter count < 1M.
    """
    def __init__(self, cfg):
        super().__init__()
        # TODO 3D conv backbone, pool spatial, project to 512, keep it small
        # 3D CNN backbone (lightweight design, params < 1M)
        self.conv3d_backbone = nn.Sequential(
            # Layer 1: downsample spatially, keep channels small
            nn.Conv3d(3, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            
            # Layer 2: further spatial downsampling
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # Layer 3: moderate channel increase
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # Layer 4: final feature extraction
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        
        # Pool out spatial dims (keep time)
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        # Temporal modeling and projection to 512 channels
        self.temporal_projection = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),  # intermediate width
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=1)  # project to 512
        )

    def forward(self, video, T_audio):
        """Same interface as EncoderA: [B, 3, Tv, 96, 96] -> [B, 512, T_audio]"""
        """
        Args:
            video:   [B, 3, T_v, 96, 96]
            T_audio: int, target audio temporal length

        Returns:
            [B, 512, T_audio]
        """
        B, C, T_v, H, W = video.shape
        
        # 3D CNN feature extraction
        # [B, 3, T_v, 96, 96] → [B, 128, T_v, h, w]
        features_3d = self.conv3d_backbone(video)
        
        # Pool spatial dims H/W
        # [B, 128, T_v, h, w] → [B, 128, T_v, 1, 1] → [B, 128, T_v]
        features = self.spatial_pool(features_3d)
        features = features.squeeze(-1).squeeze(-1)  # [B, 128, T_v]
        
        # Temporal modeling + projection
        # [B, 128, T_v] → [B, 512, T_v]
        temporal_features = self.temporal_projection(features)
        
        # Align to audio temporal length
        # [B, 512, T_v] → [B, 512, T_audio]
        output = F.interpolate(temporal_features, size=T_audio, mode='linear', align_corners=False)
        
        return output


def create_visual_encoder(cfg):
    """
    Factory function for visual encoders.

    Args:
        cfg: config dict containing cfg['visual_cfg']['encoder_type'] ('A' or 'B').

    Returns:
        An instance of LiteVisualEncoderA or LiteVisualEncoderB.
    """
    encoder_type = cfg.get('visual_cfg', {}).get('encoder_type', 'A')
    
    if encoder_type == 'A':
        return LiteVisualEncoderA(cfg)
    elif encoder_type == 'B':
        return LiteVisualEncoderB(cfg)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# Test code (optional)
if __name__ == "__main__":
    # Test configuration
    cfg = {
        'visual_cfg': {
            'encoder_type': 'A',
            'freeze_visual_encoder': True
        }
    }
    
    # Test inputs
    batch_size = 2
    T_video = 75  # 3s video, 25fps
    T_audio = 480  # 3s video, 160fps
    
    # Create encoder A
    encoder_a = LiteVisualEncoderA(cfg)
    
    # Simulated input data
    video_input = torch.randn(batch_size, 3, T_video, 96, 96)
    
    # forward propagation
    visual_raw = encoder_a(video_input, T_audio)
    
    print(f"EncoderA input shape: {video_input.shape}")
    print(f"EncoderA input shape: {visual_raw.shape}")
    
    # Check parameter freezing
    print("\nEncoderA parameter freeze check:")
    trainable_params = []
    frozen_params = []
    for name, param in encoder_a.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)
    
    print("trainable parameter:")
    for name in trainable_params:
        print(f"  - {name}")
    
    print("\nFreeze parameter:")
    for name in frozen_params:
        print(f"  - {name}")
    
    # Check parameter count
    total_params = sum(p.numel() for p in encoder_a.parameters())
    trainable_params_count = sum(p.numel() for p in encoder_a.parameters() if p.requires_grad)
    frozen_params_count = sum(p.numel() for p in encoder_a.parameters() if not p.requires_grad)
    
    print(f"\nEncoderA parametric statistics:")
    print(f"General parameter: {total_params:,}")
    print(f"Trainable parameter: {trainable_params_count:,}")
    print(f"Freeze parameter: {frozen_params_count:,}")
    
    # Test encoder B
    encoder_b = LiteVisualEncoderB(cfg)
    video_input_b = torch.randn(batch_size, 3, T_video, 96, 96)
    output_b = encoder_b(video_input_b, T_audio)
    
    print(f"\nEncoderB input shape: {video_input_b.shape}")
    print(f"EncoderB output shape: {output_b.shape}")
    
    # Check the number of encoderb parameters
    total_params_b = sum(p.numel() for p in encoder_b.parameters())
    print(f"EncoderB general parameter: {total_params_b:,}")
    
    # Check if the number of parameters is less than 1M
    if total_params_b < 1_000_000:
        print(f"[OK] EncoderB parameter count {total_params_b:,} < 1M, meets lightweight requirements")
    else:
        print(f"[FAIL] EncoderB parameter count {total_params_b:,} > 1M, Further optimization is needed")
