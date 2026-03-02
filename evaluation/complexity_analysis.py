# Shunjie works
# Model complexity comparison across variants.
# For each model variant (SEMamba, LiteAVSEMamba, and ablations): count total and trainable params, measure MACs with thop or ptflops,
# measure real-time factor (RTF) and peak GPU memory.
# Print as a table in terminal, also export to CSV and LaTeX format.

import argparse
import time
import csv
import torch
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.generator import SEMamba, LiteAVSEMamba
from utils.util import load_config


def count_parameters(model):
    """
    Count total and trainable parameters.

    Args:
        model: nn.Module
    Returns:
        (total_params: int, trainable_params: int)
    """
    # TODO
    raise NotImplementedError


def measure_macs(model, dummy_inputs, device):
    """
    Measure multiply-accumulate operations.

    Args:
        model:        nn.Module on device
        dummy_inputs: tuple of tensors matching model.forward() signature
        device:       torch.device
    Returns:
        macs: float (in GMACs)

    You can use thop, ptflops, or fvcore for this.
    """
    # TODO
    raise NotImplementedError


def measure_rtf(model, dummy_inputs, device, n_runs=50, audio_duration_sec=1.0):
    """
    Measure real-time factor: inference_time / audio_duration.
    RTF < 1.0 means faster than real-time.

    Args:
        model:              nn.Module on device, in eval mode
        dummy_inputs:       tuple of tensors
        device:             torch.device
        n_runs:             int, number of forward passes to average over
        audio_duration_sec: float, duration of the test segment
    Returns:
        rtf: float
    """
    # TODO warm up a few passes first, then time n_runs
    # GPU ops are async so you need torch.cuda.synchronize() before timing
    raise NotImplementedError


def measure_peak_gpu_memory(model, dummy_inputs, device):
    """
    Measure peak GPU memory during a forward pass.

    Args:
        model:        nn.Module on device
        dummy_inputs: tuple of tensors
        device:       torch.device
    Returns:
        peak_mb: float (megabytes)
    """
    # TODO torch has built-in CUDA memory tracking for this
    raise NotImplementedError


def build_dummy_inputs(cfg, device, include_video=False):
    """
    Create dummy tensors that match model input shapes.

    Args:
        cfg:           config dict
        device:        torch.device
        include_video: if True, also create a dummy video tensor
    Returns:
        tuple of tensors: (noisy_mag, noisy_pha) or (noisy_mag, noisy_pha, video)

    Get the shapes from the config.
    """
    # TODO
    raise NotImplementedError


def export_csv(results, output_path):
    """
    Write results list-of-dicts to CSV.

    Args:
        results:     list of dicts, each with keys like
                     'model', 'total_params', 'trainable_params', 'gmacs', 'rtf', 'peak_mb'
        output_path: str
    """
    # TODO
    raise NotImplementedError


def export_latex(results):
    """
    Print results as a LaTeX table to stdout.

    Args:
        results: same format as export_csv
    """
    # TODO
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description='Model complexity analysis')
    parser.add_argument('--config', required=True, help='path to LiteAVSE.yaml')
    parser.add_argument('--output_csv', default='evaluation/complexity.csv')
    parser.add_argument('--n_runs', type=int, default=50, help='number of runs for RTF')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO create models, run measurements, print and export results
    raise NotImplementedError


if __name__ == '__main__':
    main()
