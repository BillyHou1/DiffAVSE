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
    # STFT parameters
    n_fft = cfg['stft_cfg']['n_fft']
    hop_size = cfg['stft_cfg']['hop_size']
    win_length = cfg['stft_cfg'].get('win_length', n_fft)
    
    # Convert to tensor
    noisy_wav_tensor = torch.from_numpy(noisy_wav).float().unsqueeze(0).to(device)
    
    # Compute STFT
    noisy_mag, noisy_pha = mag_phase_stft(noisy_wav_tensor, n_fft, hop_size, win_length)
    
    # Load video if provided
    video_tensor = None
    if video_path is not None and cfg.get('lite_cfg', {}).get('use_visual', False):
        try:
            video_frames = load_video_frames(video_path, cfg)
            video_tensor = torch.from_numpy(video_frames).float().unsqueeze(0).to(device)
        except Exception as e:
            print(f"Warning: Failed to load video: {e}")
    
    # Run model
    with torch.no_grad():
        if video_tensor is not None:
            enhanced_mag, enhanced_pha = model(noisy_mag, noisy_pha, video_tensor)
        else:
            enhanced_mag, enhanced_pha = model(noisy_mag, noisy_pha)
    
    # ISTFT
    enhanced_wav = mag_phase_istft(enhanced_mag, enhanced_pha, n_fft, hop_size, win_length)
    
    return enhanced_wav.squeeze(0).cpu().numpy()


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
    # Compute STFT
    D = librosa.stft(wav, n_fft=n_fft, hop_length=hop_size)
    mag = np.abs(D)
    
    # Convert to dB scale
    mag_db = librosa.amplitude_to_db(mag, ref=np.max)
    
    # Plot
    img = librosa.display.specshow(
        mag_db, 
        sr=sr, 
        hop_length=hop_size, 
        x_axis='time', 
        y_axis='hz',
        ax=ax,
        cmap='viridis'
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    
    return img


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
    sr = cfg['stft_cfg']['sampling_rate']
    n_fft = cfg['stft_cfg']['n_fft']
    hop_size = cfg['stft_cfg']['hop_size']
    
    # Determine number of panels
    n_panels = 4 if show_diff else 3
    
    # Create figure
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    
    if n_panels == 1:
        axes = [axes]
    
    # Plot each spectrogram
    plot_spectrogram(axes[0], noisy_wav, sr, n_fft, hop_size, 'Noisy')
    plot_spectrogram(axes[1], enhanced_wav, sr, n_fft, hop_size, 'Enhanced')
    plot_spectrogram(axes[2], clean_wav, sr, n_fft, hop_size, 'Clean')
    
    # Difference heatmap
    if show_diff:
        # Compute spectrograms
        D_clean = np.abs(librosa.stft(clean_wav, n_fft=n_fft, hop_length=hop_size))
        D_enhanced = np.abs(librosa.stft(enhanced_wav, n_fft=n_fft, hop_length=hop_size))
        
        # Align lengths
        min_len = min(D_clean.shape[1], D_enhanced.shape[1])
        D_clean = D_clean[:, :min_len]
        D_enhanced = D_enhanced[:, :min_len]
        
        # Compute difference in dB
        diff = D_clean - D_enhanced
        diff_db = librosa.amplitude_to_db(np.abs(diff) + 1e-10, ref=np.max)
        
        # Plot difference
        img = librosa.display.specshow(
            diff_db,
            sr=sr,
            hop_length=hop_size,
            x_axis='time',
            y_axis='hz',
            ax=axes[3],
            cmap='RdBu_r'
        )
        axes[3].set_title('Difference (Clean - Enhanced)', fontsize=12)
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


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
