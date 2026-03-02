# LiteAVSEMamba: Technical Design Document

## 1. Motivation

SEMamba (Chao et al., SLT 2024) is an audio-only speech enhancement model that uses Mamba state-space blocks to denoise magnitude and phase spectrograms. It achieves competitive results on VCTK-DEMAND but struggles in low-SNR conditions where the audio signal alone is highly corrupted.

Our key insight: when audio is severely degraded, visual information from the speaker's face — particularly lip movements — provides a complementary signal that is independent of acoustic noise. We extend SEMamba into **LiteAVSEMamba**, an audio-visual model with three design goals:

1. **Effective fusion** — visual information should help when it's reliable, but not hurt when it's not (e.g., occluded face, poor video quality)
2. **Causal processing** — the model should support real-time / streaming inference (no future lookahead in the time dimension)
3. **Lightweight additions** — the visual branch should add minimal parameters on top of SEMamba's ~10.5M baseline

---

## 2. Architecture Overview

```
Input:  noisy_mag [B, F, T]    noisy_pha [B, F, T]    video [B, 3, Tv, 96, 96]
         │                        │                       │
         ▼                        ▼                       ▼
    ┌─────────────────────┐              ┌──────────────────────┐
    │   DenseEncoder      │              │  Visual Encoder      │
    │   [B,2,T,F] →       │              │  (MobileNetV3-Small) │
    │   [B,C,T,F_enc]     │              │  [B,3,Tv,96,96] →   │
    └─────────┬───────────┘              │  [B,512,T_audio]    │
              │                          └──────────┬───────────┘
              │                                     │
              │                          ┌──────────▼───────────┐
              │                          │  VCE (Confidence)    │
              │                          │  [B,T,512] →         │
              │                          │  alpha [B,T,1]       │
              │                          └──────────┬───────────┘
              │                                     │
              │    ┌────────────────────────────────┐│
              │    │  Visual Projection             ││
              │    │  512 → C, then expand to       ││
              │    │  [B, C, T, F_enc]              ││
              │    └───────────────┬────────────────┘│
              │                   │                  │
              ▼                   ▼                  │
        ┌─────────────────────────────┐              │
        │  FSVG (Freq-Selective Gate) │              │
        │  audio_feat + visual_feat → │              │
        │  gate [B, 1, T, F_enc]     │              │
        └─────────────┬──────────────┘              │
                      │                              │
              ┌───────▼──────────────────────────────▼──┐
              │  Double-Gated Fusion:                     │
              │  fused = audio + alpha * gate * visual   │
              └───────────────┬──────────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  CausalTFMambaBlock  │
                   │  × N (default 4)    │
                   └──────────┬──────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            ┌──────────────┐   ┌──────────────┐
            │  MagDecoder  │   │ PhaseDecoder │
            └──────┬───────┘   └──────┬───────┘
                   ▼                   ▼
            denoised_mag         denoised_pha         denoised_com
            [B, F, T]            [B, F, T]            [B, F, T, 2]
```

---

## 3. Signal Flow in Detail

### 3.1 Audio Path (inherited from SEMamba)

The audio path is largely identical to SEMamba:

1. **STFT**: 16 kHz audio → `n_fft=400, hop_size=100, win_size=400` → magnitude and phase spectrograms of shape `[B, 201, T]` (201 frequency bins = n_fft/2 + 1). Power compression with factor 0.3 is applied to magnitude.

2. **DenseEncoder**: Takes concatenated `[noisy_mag, noisy_pha]` as `[B, 2, T, F]`, outputs `[B, C, T, F_enc]` where `C=64` (hidden feature dim) and `F_enc=100` (frequency dimension after stride-2 convolution: `(201-3)/2 + 1 = 100`).

3. **Mamba Blocks**: N=4 CausalTFMambaBlock layers process the encoded features. Each block scans along time (causal, forward-only) and frequency (bidirectional) using Mamba state-space models.

4. **Decoders**: MagDecoder produces a multiplicative mask via LearnableSigmoid2D, applied to the input magnitude. PhaseDecoder predicts phase directly via `atan2(imag, real)`.

5. **iSTFT**: Reconstructed magnitude and phase → complex spectrogram → inverse STFT → enhanced waveform.

### 3.2 Visual Path (new)

1. **Visual Encoder** (`LiteVisualEncoderA`): Frozen MobileNetV3-Small backbone extracts per-frame features from 96×96 face crops. A temporal Conv1d layer models lip dynamics across frames. Output is projected to `[B, 512, T_audio]` using `F.interpolate` to match the audio frame rate (25 fps video → 160 fps STFT frames).

2. **VCE — Visual Confidence Estimator**: A small MLP that scores each frame's reliability: `[B, T, 512] → [B, T, 1]` with output in `[0, 1]` via sigmoid. When the face is clear and lip movement is informative, alpha → 1. When the face is occluded, blurry, or uninformative, alpha → 0. This is learned end-to-end — the model discovers what "reliable" means through the training signal.

3. **Visual Projection**: Linear layer maps 512-dim visual features to C=64 channels, then the temporal dimension is expanded across frequency bins to get `[B, C, T, F_enc]`, matching the audio feature shape.

4. **FSVG — Frequency-Selective Visual Gating**: Takes audio features `[B, C, T, F_enc]` and visual features `[B, C, T, F_enc]`, concatenates along channel dim to `[B, 2C, T, F_enc]`, passes through a lightweight Conv2d network, outputs `gate ∈ [0, 1]` of shape `[B, 1, T, F_enc]`. The gate learns that speech frequencies (300 Hz–3 kHz) benefit most from visual information, while high-frequency noise regions should receive less visual influence.

### 3.3 Double-Gated Fusion

The core fusion equation:

```
fused = audio_feat + alpha * gate * visual_feat
```

Where:
- `audio_feat`: `[B, C, T, F_enc]` from DenseEncoder
- `visual_feat`: `[B, C, T, F_enc]` projected visual features
- `alpha`: `[B, T, 1]` → broadcast to `[B, 1, T, 1]` — per-frame confidence
- `gate`: `[B, 1, T, F_enc]` — per-frequency gating weight

This is a **residual design**: when `alpha=0` (bad video) or `gate=0` (irrelevant frequency), the model falls back to pure audio features, exactly recovering SEMamba behaviour. This guarantees that adding vision can only help, never hurt — a critical property for robustness.

---

## 4. Key Design Decisions

### 4.1 Why Double Gating (VCE + FSVG) Instead of Simple Concatenation?

Simple concatenation or addition of audio and visual features is the baseline approach (used in many AV-SE papers). The problem is that it gives the visual stream equal influence regardless of video quality or frequency relevance.

Our double-gating addresses two orthogonal dimensions:

| Dimension | Module | What it controls |
|-----------|--------|-----------------|
| **Temporal** | VCE (alpha) | "Is this video frame reliable?" — handles occlusion, motion blur, low quality |
| **Spectral** | FSVG (gate) | "Does this frequency bin benefit from visual info?" — lip movements correlate with 300 Hz–3 kHz, not with high-freq noise |

The multiplicative interaction `alpha * gate` means both conditions must be met for visual information to flow through. This is inspired by gated multimodal units (Arevalo et al., 2017) and FiLM conditioning (Perez et al., 2018), adapted to the speech enhancement T-F domain.

### 4.2 Why Causal Mamba Instead of Bidirectional?

SEMamba's original TFMambaBlock uses bidirectional Mamba scanning in both time and frequency. This requires the full utterance before processing — fine for offline enhancement but unsuitable for real-time applications.

Our CausalTFMambaBlock makes the **time dimension causal** (forward-only) while keeping **frequency bidirectional**:

- **Time (causal)**: A real-time system must process audio frame-by-frame as it arrives. The Mamba block only sees current and past time steps, enabling streaming.
- **Frequency (bidirectional)**: All frequency bins at a given time instant exist simultaneously — there is no temporal ordering in frequency. Bidirectional scanning allows the model to use both low and high frequency context.

This is one of the key novelties over the original SEMamba. The architectural change in MambaBlock:
- `bidirectional=True`: forward + backward pass → output `[B, T, 2C]` → ConvTranspose1d to project back to `[B, T, C]`
- `bidirectional=False`: forward pass only → output `[B, T, C]` → ConvTranspose1d from C to C (or skip)

### 4.3 Why Frozen MobileNetV3-Small?

We need a visual encoder that:
1. Extracts meaningful facial features from 96×96 crops
2. Is pretrained (we don't have enough data to train a vision model from scratch)
3. Is small enough not to dominate the parameter budget

MobileNetV3-Small (Howard et al., 2019) is ~2.5M parameters. Frozen, it adds **zero trainable parameters** to the model. Only the temporal Conv1d and projection layers are trained. The exact trainable parameter count depends on the temporal modelling design, but is targeted to stay well under 1M to keep total model size close to the SEMamba baseline.

We also provide EncoderB (custom 3D CNN, <1M params, trains from scratch) as an ablation alternative.

### 4.4 Why Residual Fusion After Encoder, Not at Input?

We fuse visual features **after DenseEncoder in feature space**, not at the raw spectrogram level. Reasons:

1. **Dimensionality**: Raw spectrograms have F=201 bins; after encoding, F_enc=100. Fusing in feature space is cheaper.
2. **Abstraction level**: DenseEncoder transforms raw spectrograms into learned representations where audio and visual features are more comparable.
3. **Preserving audio path**: The input_channel stays 2 (mag + pha), identical to SEMamba. The DenseEncoder weights can be initialized from a pretrained SEMamba checkpoint.
4. **Clean fallback**: When video=None, the entire visual branch is skipped and the model is functionally identical to SEMamba with CausalTFMambaBlock.

### 4.5 Audio-Only Fallback

When `video=None` is passed to `LiteAVSEMamba.forward()`:
- Visual encoder, VCE, FSVG are all skipped
- Fusion reduces to `fused = audio_feat + 0 = audio_feat`
- The rest of the pipeline (CausalTFMamba → decoders) runs normally

This means a single model checkpoint handles both AV and audio-only inference. During training, we can optionally drop video with some probability to make the model robust to missing video at test time.

---

## 5. Training Strategy

### 5.1 Loss Functions

We use a composite loss with 5+1 terms (weights from config):

| Loss | Weight | Formula | Purpose |
|------|--------|---------|---------|
| Magnitude | 0.9 | L1(clean_mag, denoised_mag) | Spectral envelope accuracy |
| Phase | 0.3 | Anti-wrapping phase loss (IP + GD + IAF) | Phase reconstruction |
| Complex | 0.1 | MSE(clean_com, denoised_com) × 2 | Joint mag-phase consistency |
| Consistency | 0.1 | MSE(denoised_com, STFT(iSTFT(denoised))) × 2 | STFT consistency constraint |
| SI-SDR | 0.3 | -SI-SDR(clean_wav, enhanced_wav) | Time-domain perceptual quality |
| Time | 0.0 | L1(clean_wav, enhanced_wav) | Disabled |

No discriminator — generator-only training with AdamW optimizer.

### 5.2 Data Pipeline

- **Training data**: GRID corpus (28 speakers train, 3 valid, 3 test) for prototyping; LRS2 (224 hours) for full training
- **Noise sources**: DEMAND (16 types) + DNS Challenge (63K files) + MUSAN
- **SNR range**: [-5, 20] dB, uniformly sampled per sample
- **Augmentation**: RIR convolution (30% probability), visual degradation (random dropout/blur/blackout for VCE training)
- **Segment size**: 16000 samples (1 second) → 160 STFT frames, 25 video frames

### 5.3 Visual Augmentation for VCE

To train VCE effectively, we augment video quality during training:
- ~60% original (clean video)
- ~8% all-black frames (simulates complete occlusion)
- ~10% random frame dropout
- ~10% Gaussian noise
- ~7% blur
- ~5% dim brightness

This forces VCE to learn meaningful confidence scores — it must output low alpha for degraded video and high alpha for clean video, purely from the training signal.

---

## 6. Comparison with SEMamba

| Aspect | SEMamba | LiteAVSEMamba |
|--------|---------|---------------|
| Input | Audio only (mag + pha) | Audio + Video (96×96 face) |
| Mamba blocks | Bidirectional (TFMambaBlock) | Causal time + Bidirectional freq (CausalTFMambaBlock) |
| Encoder | DenseEncoder | DenseEncoder (same) + Visual Encoder |
| Fusion | None | Double-gated: alpha (VCE) × gate (FSVG) × visual |
| Fallback | N/A | Graceful degradation to audio-only when video=None |
| Parameters | ~10.5M | ~10.5M + <1M trainable visual params (encoder frozen) |
| Streaming | No (bidirectional) | Yes (causal time) |
| Training | GAN-based (generator + discriminator) | Generator-only with SI-SDR loss |

---

## 7. Implementation Status

| Component | File | Owner | Status |
|-----------|------|-------|--------|
| SEMamba (baseline) | generator.py | (original) | Done |
| LiteAVSEMamba | generator.py | Billy | Skeleton ready, TODO implementation |
| VCE | vce.py | Billy | Skeleton ready |
| FSVG | fsvg.py | Dominic | Done |
| CausalTFMambaBlock | mamba_block.py | Dominic | Skeleton ready |
| Visual Encoder | lite_visual_encoder.py | Zhenning | Skeleton ready |
| Lite Codec | codec_module.py | Zhenning | Skeleton ready |
| AV Dataloader | dataloader_av.py | Fan + Zhenning | Audio part done (under review), video part TODO |
| Data scripts | data/*.py | Fan | Done (under review) |
| Loss functions | loss.py | Ronny + Shunjie | Existing losses done, SI-SDR and STOI TODO |
| Training loop | train_lite.py | Ronny | Skeleton ready |
| Evaluation | evaluation/*.py | Fan + Shunjie | Skeletons ready |

All interface contracts (input/output shapes, config keys) are finalized. W3 integration target: end-to-end forward pass working.

---

## 8. Datasets

| Dataset | Purpose | Size | Status |
|---------|---------|------|--------|
| GRID | Prototyping & ablation (33 speakers, 28/3/3 split) | ~10 GB | On HPC |
| LRS2 | Full training (224 hours, multi-speaker) | ~50 GB | Downloaded, pending transfer |
| DEMAND | Noise source (16 environmental types) | ~5 GB | On HPC |
| DNS Challenge | Noise source (63K files, diverse) | ~59 GB | On HPC |
| MUSAN | Noise source (music, speech, noise) | ~12 GB | On HPC |
| RIRS_NOISES | Room impulse responses for augmentation | ~4 GB | On HPC |
