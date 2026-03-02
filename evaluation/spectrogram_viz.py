# Shunjie
# Spectrogram visualization.
# Generate a 3-panel figure: noisy / enhanced / clean spectrograms side by side.
# Optionally add a 4th panel with the difference heatmap (clean - enhanced).
# Use librosa.display or matplotlib directly. Save output as PNG.

import argparse
import os
import numpy as np
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.generator import LiteAVSEMamba
from models.stfts import mag_phase_stft, mag_phase_istft
from dataloaders.dataloader_av import load_video_frames
from utils.util import load_config, load_checkpoint


def enhance_audio(model, noisy_wav, video_path, cfg, device):
    """
    Run model inference on a single waveform.
    Same pipeline as inference_av.py — reuse or adapt from there.

    Args:
        model:      LiteAVSEMamba in eval mode
        noisy_wav:  1-D numpy array
        video_path: str or None
        cfg:        config dict
        device:     torch.device
    Returns:
        enhanced_wav: 1-D numpy array
    """
    # TODO
    raise NotImplementedError


def plot_spectrogram(ax, wav, sr, n_fft, hop_size, title):
    """
    Plot a single spectrogram on the given axes.

    Args:
        ax:       matplotlib Axes
        wav:      1-D numpy array
        sr:       int, sampling rate
        n_fft:    int
        hop_size: int
        title:    str
    """
    # TODO magnitude spectrogram in dB, display with librosa or matplotlib
    raise NotImplementedError


def generate_figure(noisy_wav, enhanced_wav, clean_wav, cfg, output_path,
                    show_diff=True):
    """
    Generate a multi-panel spectrogram comparison figure.

    Args:
        noisy_wav:    1-D numpy array
        enhanced_wav: 1-D numpy array
        clean_wav:    1-D numpy array
        cfg:          config dict
        output_path:  str, path to save PNG
        show_diff:    bool, whether to add 4th panel with difference heatmap

    Panels: Noisy | Enhanced | Clean | (optional) Difference heatmap
    """
    # TODO multi-panel figure, call plot_spectrogram for each,
    # add difference heatmap if show_diff, save to output_path
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description='Spectrogram visualization')
    parser.add_argument('--config', required=True, help='path to LiteAVSE.yaml')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--clean_audio', required=True, help='path to clean .wav')
    parser.add_argument('--noisy_audio', required=True, help='path to noisy .wav')
    parser.add_argument('--video', default=None, help='path to video (optional)')
    parser.add_argument('--output', default='evaluation/spectrogram.png')
    parser.add_argument('--no_diff', action='store_true', help='skip difference panel')
    args = parser.parse_args()

    cfg = load_config(args.config)
    sr = cfg['stft_cfg']['sampling_rate']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = LiteAVSEMamba(cfg).to(device)
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(state_dict['generator'])
    model.eval()

    # Load audio
    clean_wav, _ = librosa.load(args.clean_audio, sr=sr)
    noisy_wav, _ = librosa.load(args.noisy_audio, sr=sr)

    # Enhance
    with torch.no_grad():
        enhanced_wav = enhance_audio(model, noisy_wav, args.video, cfg, device)

    # Plot
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    generate_figure(noisy_wav, enhanced_wav, clean_wav, cfg, args.output,
                    show_diff=not args.no_diff)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
