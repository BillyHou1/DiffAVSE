# LiteAVSEMamba

Lightweight audio-visual speech enhancement with Mamba. We take SEMamba [1] and add visual information from face video so the model can use lip movements to help with denoising.

University of Bristol, Feb 2026

This branch builds on [SEMamba](https://github.com/RoyChao19477/SEMamba) by Chao et al. All new files have TODO skeletons, open your assigned file and check the header comments.

### Team

| | Role | Scope |
|---|---|---|
| P1 Billy | System Architect | architecture design, VCE, integration, inference |
| P2 Dominic | Fusion & Demo Lead | FSVG, Mamba modification, web demo |
| P3 Zhenning | Backbone & Data Lead | visual encoder, lite codec, video pipeline |
| P4 Fan | Audio Data Lead | data scripts, audio pipeline, noise mixing, SNR analysis |
| P5 Ronny | Training Lead | SI-SDR loss, training loop, all training experiments |
| P6 Shunjie | Eval & Bench Lead | STOI metric, complexity analysis, figures, statistical tests |

---

## What we're doing

SEMamba is audio-only. We're adding a visual stream so the model can see the speaker's face and use that to denoise better, especially in heavy noise. The key idea is a double-gated fusion:

```
fused = audio_feat + alpha * gate * visual_feat
```

`alpha` comes from VCE, a small MLP that looks at the visual features and decides how reliable the video is. If the face is occluded or the video quality is bad, alpha goes toward zero and the model falls back to audio-only. `gate` comes from FSVG, a conv-based module that learns per-frequency weights, so speech frequencies around 300 Hz to 3 kHz get more visual influence than high frequencies where video doesn't really help.

We use Mamba blocks instead of transformers because they run in linear time, which matters if we want this to work in real-time later. The original SEMamba Mamba blocks are bidirectional, so we also add a causal variant for streaming.

Visual input is 96x96 RGB face crops at 25fps, audio is 16kHz mono. The visual encoder is a frozen MobileNetV3-Small so we don't need to train a huge vision model from scratch. The SEMamba baseline is about 10.5M parameters and we're trying to keep our additions small.

---

## Architecture

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/349352a5-6164-4377-becf-81902935bff9" />

---

## Files

The original SEMamba files are all still here untouched. We added new files for the visual pipeline and modified a few existing ones to support the AV version.

```
LiteAVSEMamba/
|
|-- models/
|   |-- generator.py              SEMamba class unchanged, LiteAVSEMamba is new (Billy)
|   |-- mamba_block.py            original blocks + new CausalTFMambaBlock (Dominic)
|   |-- codec_module.py           original encoder/decoder + lite depthwise-separable versions (Zhenning)
|   |-- vce.py                    new, Visual Confidence Estimator (Billy)
|   |-- fsvg.py                   new, Frequency-Selective Visual Gating (Dominic)
|   |-- lite_visual_encoder.py    new, MobileNetV3 per-frame + 3D CNN temporal (Zhenning)
|   |-- loss.py                   added si_sdr_loss (Ronny) and stoi_score (Shunjie)
|   |-- stfts.py                  unchanged
|   |-- lsigmoid.py              unchanged
|   |-- discriminator.py          unchanged, we don't use it
|   +-- pcs400.py                 unchanged, we don't use it
|
|-- dataloaders/
|   |-- dataloader_vctk.py        unchanged, original VCTK-DEMAND loader
|   |-- dataloader_av.py          new, Fan (audio) + Zhenning (video)
|   +-- augment_rir.py            new, room impulse response augmentation (Fan)
|
|-- data/
|   |-- make_grid_json.py         new, walks GRID corpus and outputs JSON lists (Fan)
|   |-- make_lrs_json.py          same for LRS2 (Fan)
|   |-- make_vox_json.py          same for VoxCeleb2 (Fan)
|   |-- prepare_noise.py          new, merges noise sources into one pool (Fan)
|   +-- make_dataset_json.py      unchanged, original VCTK path collector
|
|-- recipes/
|   |-- SEMamba_advanced/          unchanged
|   |-- SEMamba_advanced_PCS/      unchanged
|   +-- LiteAVSE/
|       +-- LiteAVSE.yaml         our config
|
|-- evaluation/
|   |-- snr_breakdown.py          per-SNR metric breakdown (Fan)
|   |-- complexity_analysis.py    params/MACs/RTF comparison (Shunjie)
|   |-- spectrogram_viz.py        spectrogram figures (Shunjie)
|   +-- statistical_test.py       significance tests (Shunjie)
|
|-- train.py                      unchanged, SEMamba audio-only training with GAN
|-- train_lite.py                 our training loop, no discriminator (Ronny + Fan + Billy)
|-- inference.py                  unchanged, SEMamba inference
|-- inference_av.py               our AV inference (Billy)
+-- utils/util.py                 added NaN safety helpers to the existing utils
```

---

## Getting started

Read the SEMamba paper [1] first to understand the base architecture. If you're working on mamba_block.py, also read the Mamba paper [2]. Then open your assigned file and read the TODO comments at the top. Look at the existing SEMamba code like train.py and generator.py to see how things are structured, our code follows the same patterns.

```bash
git clone -b dev_LiteAVSE https://github.com/BillyHou1/LiteAVSE.git
cd LiteAVSE
pip install -r requirements.txt
# install mamba-ssm separately
```

### Datasets

For speech we have two options. [GRID](https://spandh.dcs.shef.ac.uk/gridcorpus/) is about 28 hours of single-speaker data, it's public so use it for prototyping and ablations. [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) is much bigger at 224 hours with multiple speakers, but you need to apply for access. We'll use that for full training.

For noise, [DEMAND](https://zenodo.org/record/1227121) only has 18 noise types which isn't enough on its own. Combine it with [DNS Challenge](https://github.com/microsoft/DNS-Challenge) or [MUSAN](https://www.openslr.org/17/) for better coverage.

Run the scripts in `data/` to make the JSON lists the dataloader expects.

### Training

```bash
# audio-only baseline, this already works
python train.py --config recipes/SEMamba_advanced/SEMamba_advanced.yaml

# audio-visual, needs all TODOs done first
python train_lite.py --config recipes/LiteAVSE/LiteAVSE.yaml \
    --exp_folder exp --exp_name LiteAVSE_v1
```

---

## Timeline

Feb 10 - Apr 28, 2026. Key dates: W3 integration (Mar 2), W6 code freeze (Mar 22), W10 internal deadline (Apr 21), submission (Apr 28).

### W1 (Feb 10-16) SETUP

Everyone gets the environment running. Reproduce SEMamba baseline and verify PESQ matches the paper. Download GRID and DEMAND. Skeleton files and interface specs finalized.

### W2 (Feb 17-23) BUILD

Everyone writes their module in parallel, nothing depends on anything else yet.

| File | Who | What to do |
|------|-----|------------|
| `data/make_grid_json.py` | Fan | Walk GRID corpus, pair .wav + .mpg, split by speaker 90/5/5 |
| `data/make_lrs_json.py` | Fan | Walk LRS2, use official split files, output JSON lists |
| `data/make_vox_json.py` | Fan | Walk VoxCeleb2, hold out 2-3% of dev as validation |
| `data/prepare_noise.py` | Fan | Merge DEMAND + DNS + MUSAN into one noise pool, split train/val |
| `dataloaders/augment_rir.py` | Fan | RIR augmentation: convolve with room impulse responses, energy normalize |
| `dataloaders/dataloader_av.py` | Fan + Zhenning | Fan: mix_audio, noise loading, STFT pipeline; Zhenning: load_video_frames, visual augmentation |
| `models/vce.py` | Billy | VCE confidence scorer + VCEWithTemporalSmoothing with causal smoothing |
| `models/fsvg.py` | Dominic | FSVG gating network + FSVGWithPrior with learnable freq prior |
| `models/lite_visual_encoder.py` | Zhenning | EncoderA pretrained backbone + EncoderB custom 3D CNN |
| `models/mamba_block.py` | Dominic | Fix mamba-ssm 2.3.0 imports, add bidirectional param, Mamba-2 switch, CausalTFMambaBlock |
| `models/codec_module.py` | Zhenning | DepthwiseSeparableConv2d, LiteDenseBlock, LiteDenseEncoder, LiteMagDecoder, LitePhaseDecoder |
| `models/loss.py` | Ronny + Shunjie | Ronny: si_sdr_loss, si_sdr_score; Shunjie: stoi_score |

### W3 (Feb 24-Mar 2) INTEGRATE

All modules are wired together in `models/generator.py`: the LiteAVSEMamba class that connects all the modules, does the double-gated fusion, and handles the case where video is None so it falls back to audio-only. End-to-end forward pass should work by end of this week. Everyone tests their own module in the integrated pipeline.

### W4 (Mar 3-9) TRAIN

Ronny gets `train_lite.py` running on GRID with a small subset first. The loop uses 5+1 losses, validates with PESQ/STOI/SI-SDR, and has NaN safety so training doesn't crash on bad batches. Team helps debug integration issues. Goal: training converges on the subset.

### W5 (Mar 10-16) TRAIN

Ronny runs full training + ablation variants in parallel: full model, w/o VCE, w/o FSVG. Fan helps with data pipeline and HPC job management. By end of week we should have trained checkpoints for all variants.

### W6 (Mar 17-22) FREEZE

Code freeze. Shunjie runs the full evaluation pipeline: complexity_analysis.py for params/MACs/RTF, snr_breakdown.py at -5/0/5/10/15/20 dB (with Fan), statistical_test.py for significance. Team reviews all results together.

### W7-W8 (Mar 23-Apr 5) WRITING

Spring break. Everyone writes their section of the report/paper. Target: complete draft with all figures by end of W8.

**Remaining code tasks:**

| File | Who | What to do |
|------|-----|------------|
| `inference_av.py` | Billy | Single-file and folder-mode inference, RMS normalize/denormalize |
| `evaluation/snr_breakdown.py` | Fan | Run model at -5/0/5/10/15/20 dB, output CSV + plots |
| `evaluation/complexity_analysis.py` | Shunjie | Params, MACs, RTF, peak GPU memory, compare variants |
| `evaluation/spectrogram_viz.py` | Shunjie | 3-panel spectrogram noisy/enhanced/clean + difference heatmap |
| `evaluation/statistical_test.py` | Shunjie | Paired t-test, Wilcoxon signed-rank, Cohen's d |

**Report sections:**

| Section | Who |
|---------|-----|
| Abstract & Introduction | Billy |
| Related Work | Zhenning |
| Method: Visual Encoder & FSVG | Dominic |
| Method: Mamba Backbone & Codec | Zhenning |
| Method: VCE & Fusion Pipeline | Billy |
| Data & Experimental Setup | Fan |
| Training Details | Ronny |
| Results & Analysis | Ronny + Shunjie |
| Complexity & Ablation Tables | Shunjie |
| Figures & Spectrograms | Shunjie |
| Demo & Conclusion | Dominic |

### W9 (Apr 6-12) DEMO

Dominic and Billy build the Gradio web demo: upload noisy audio + face video, run inference, play back enhanced audio with before/after spectrograms. Record a demo video.

### W10-W11 (Apr 13-28) FINAL

Paper finalization, cross-review, format standardization. W11 is buffer for contingencies. Submission Apr 28.

---

## References

[1] R. Chao et al., "An Investigation of Incorporating Mamba for Speech Enhancement," IEEE SLT, 2024.

[2] A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," COLM, 2024.

[3] T. A. Ma et al., "Real-Time Audio-Visual Speech Enhancement Using Pre-trained Visual Representations," Interspeech, 2025.

[4] Y.-X. Lu et al., "MP-SENet: A Speech Enhancement Model with Parallel Denoising of Magnitude and Phase Spectra," Interspeech, 2023.

[5] K. Li et al., "An Audio-Visual Speech Separation Model Inspired by Cortico-Thalamo-Cortical Circuits," IEEE TPAMI, 2024.

[6] P. Ma et al., "Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels," IEEE ICASSP, 2023.

[7] A. Howard et al., "Searching for MobileNetV3," IEEE/CVF ICCV, 2019.

[8] G. Huang et al., "Densely Connected Convolutional Networks," IEEE CVPR, 2017.

[9] J. Le Roux et al., "SDR -- Half-baked or Well Done?" IEEE ICASSP, 2019.

[10] B. Shi et al., "Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction," ICLR, 2022.

[11] C. H. Taal et al., "An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech," IEEE TASLP, 2011.

[12] D. S. Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition," Interspeech, 2019.

[13] E. Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer," AAAI, 2018.

[14] J. Arevalo et al., "Gated Multimodal Units for Information Fusion," ICLR Workshop, 2017.

[15] A. W. Rix et al., "Perceptual Evaluation of Speech Quality (PESQ)," IEEE ICASSP, 2001.

---

## License

Based on [SEMamba](https://github.com/RoyChao19477/SEMamba) by R. Chao et al.

