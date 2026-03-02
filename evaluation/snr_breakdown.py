# Fan
# Per-SNR evaluation breakdown.
# Run the model on test set at fixed SNR levels: -5, 0, 5, 10, 15, 20 dB.
# Compute PESQ, STOI, and SI-SDR at each level separately.
# Optionally break down by noise type if noise labels are available.
# Save results to CSV and generate line plots (metric vs SNR).

import argparse
import os
import csv
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.generator import LiteAVSEMamba
from models.stfts import mag_phase_stft, mag_phase_istft
from models.loss import pesq_score, si_sdr_score, stoi_score
from dataloaders.dataloader_av import load_json_file, load_video_frames, mix_audio
from utils.util import load_config, load_checkpoint

SNR_LEVELS = [-5, 0, 5, 10, 15, 20]


def evaluate_at_snr(model, clean_path, noise_path, video_path, snr_db, cfg, device):
    """
    Enhance one utterance at a specific SNR and return per-sample metrics.

    Args:
        model:      LiteAVSEMamba in eval mode
        clean_path: str, path to clean .wav
        noise_path: str, path to noise .wav
        video_path: str or None, path to .mp4/.mpg
        snr_db:     float
        cfg:        config dict
        device:     torch.device
    Returns:
        dict with keys 'pesq', 'stoi', 'si_sdr' (float values)
    """
    # TODO mix at the given SNR, run through model, compute metrics
    raise NotImplementedError


def run_snr_breakdown(model, test_data, noise_list, cfg, device):
    """
    Run evaluation at each SNR level.

    Args:
        model:      LiteAVSEMamba in eval mode
        test_data:  list of dicts from test JSON (each has 'audio', optionally 'video')
        noise_list: list of noise file paths
        cfg:        config dict
        device:     torch.device
    Returns:
        dict mapping snr_db -> {'pesq': mean, 'stoi': mean, 'si_sdr': mean}
    """
    # TODO loop over SNR_LEVELS, evaluate all samples at each, average
    raise NotImplementedError


def save_csv(results, output_path):
    """
    Save per-SNR results to CSV.

    Args:
        results:     dict mapping snr_db -> {'pesq': float, 'stoi': float, 'si_sdr': float}
        output_path: str
    """
    # TODO header + one row per SNR level
    raise NotImplementedError


def plot_results(results, output_path):
    """
    Generate line plots: one subplot per metric, x-axis = SNR, y-axis = score.

    Args:
        results:     dict mapping snr_db -> {'pesq': float, 'stoi': float, 'si_sdr': float}
        output_path: str, path to save PNG
    """
    # TODO 3 subplots side by side, one per metric, save as PNG
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description='Per-SNR evaluation breakdown')
    parser.add_argument('--config', required=True, help='path to LiteAVSE.yaml')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--test_json', default=None, help='override test data json path')
    parser.add_argument('--noise_json', default=None, help='override noise json path')
    parser.add_argument('--output_dir', default='evaluation/snr_results')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = LiteAVSEMamba(cfg).to(device)
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(state_dict['generator'])
    model.eval()

    # Load data lists
    test_json = args.test_json or cfg['data_cfg']['test_data_json']
    noise_json = args.noise_json or cfg['data_cfg']['valid_noise_json']
    test_data = load_json_file(test_json)
    noise_list = load_json_file(noise_json)

    os.makedirs(args.output_dir, exist_ok=True)

    # TODO run evaluation, save csv, plot
    raise NotImplementedError


if __name__ == '__main__':
    main()
