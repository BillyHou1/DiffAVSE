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
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


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
    try:
        from thop import profile
        macs, _ = profile(model, inputs=dummy_inputs, verbose=False)
        return macs / 1e9  # Convert to GMACs
    except ImportError:
        # Fallback: estimate from params and input size
        total_params, _ = count_parameters(model)
        # Rough estimate: each param contributes ~2 MACs per input element
        input_elements = sum(t.numel() for t in dummy_inputs)
        estimated_macs = total_params * 2 * input_elements
        return estimated_macs / 1e9


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
    model.eval()
    
    # Warm up
    with torch.no_grad():
        for _ in range(5):
            _ = model(*dummy_inputs)
    
    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Time n_runs
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(*dummy_inputs)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_inference_time = total_time / n_runs
    rtf = avg_inference_time / audio_duration_sec
    
    return rtf


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
    if device.type != 'cuda':
        return 0.0
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats(device)
    
    # Run forward pass
    with torch.no_grad():
        _ = model(*dummy_inputs)
    
    # Get peak memory
    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mb = peak_bytes / (1024 ** 2)
    
    return peak_mb


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
    batch_size = 1
    n_fft = cfg['stft_cfg']['n_fft']
    hop_size = cfg['stft_cfg']['hop_size']
    sampling_rate = cfg['stft_cfg']['sampling_rate']
    audio_duration = 1.0  # 1 second of audio
    
    # Calculate number of frames for 1 second audio
    num_samples = int(sampling_rate * audio_duration)
    num_frames = (num_samples - n_fft) // hop_size + 1
    freq_bins = n_fft // 2 + 1
    
    # Create dummy magnitude and phase
    noisy_mag = torch.randn(batch_size, 1, num_frames, freq_bins, device=device)
    noisy_pha = torch.randn(batch_size, 1, num_frames, freq_bins, device=device)
    
    if include_video:
        # Create dummy video tensor [B, 3, T_v, H, W]
        video_fps = 25
        T_v = int(audio_duration * video_fps)
        H = W = 96
        video = torch.randn(batch_size, 3, T_v, H, W, device=device)
        return (noisy_mag, noisy_pha, video)
    
    return (noisy_mag, noisy_pha)


def export_csv(results, output_path):
    """
    Write results list-of-dicts to CSV.

    Args:
        results:     list of dicts, each with keys like
                     'model', 'total_params', 'trainable_params', 'gmacs', 'rtf', 'peak_mb'
        output_path: str
    """
    if not results:
        return
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    fieldnames = list(results[0].keys())
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"CSV exported to {output_path}")


def export_latex(results):
    """
    Print results as a LaTeX table to stdout.

    Args:
        results: same format as export_csv
    """
    if not results:
        return
    
    print("\n% LaTeX table")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lrrrrr}")
    print("\\hline")
    print("Model & Params (M) & GMACs & RTF & Peak Mem (MB) \\\\")
    print("\\hline")
    
    for r in results:
        model_name = r.get('model', 'Unknown')
        total_params = r.get('total_params', 0) / 1e6  # Convert to millions
        gmacs = r.get('gmacs', 0)
        rtf = r.get('rtf', 0)
        peak_mb = r.get('peak_mb', 0)
        
        print(f"{model_name} & {total_params:.2f} & {gmacs:.2f} & {rtf:.4f} & {peak_mb:.1f} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Model complexity comparison}")
    print("\\label{tab:complexity}")
    print("\\end{table}")


def main():
    parser = argparse.ArgumentParser(description='Model complexity analysis')
    parser.add_argument('--config', required=True, help='path to LiteAVSE.yaml')
    parser.add_argument('--output_csv', default='evaluation/complexity.csv')
    parser.add_argument('--n_runs', type=int, default=50, help='number of runs for RTF')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = []
    
    # Model variants to evaluate
    model_variants = [
        ('SEMamba', SEMamba, False),
        ('LiteAVSEMamba', LiteAVSEMamba, True),
    ]
    
    for model_name, model_class, use_video in model_variants:
        print(f"\nEvaluating {model_name}...")
        
        # Create model
        model = model_class(cfg).to(device)
        model.eval()
        
        # Build dummy inputs
        dummy_inputs = build_dummy_inputs(cfg, device, include_video=use_video)
        
        # Measure metrics
        total_params, trainable_params = count_parameters(model)
        gmacs = measure_macs(model, dummy_inputs, device)
        rtf = measure_rtf(model, dummy_inputs, device, n_runs=args.n_runs)
        peak_mb = measure_peak_gpu_memory(model, dummy_inputs, device)
        
        results.append({
            'model': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'gmacs': gmacs,
            'rtf': rtf,
            'peak_mb': peak_mb
        })
        
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  GMACs: {gmacs:.2f}")
        print(f"  RTF: {rtf:.4f}")
        print(f"  Peak GPU Memory: {peak_mb:.1f} MB")
    
    # Print summary table
    print("\n" + "="*80)
    print("COMPLEXITY SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Params (M)':<12} {'GMACs':<10} {'RTF':<10} {'Peak MB':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['model']:<20} {r['total_params']/1e6:<12.2f} {r['gmacs']:<10.2f} {r['rtf']:<10.4f} {r['peak_mb']:<10.1f}")
    print("="*80)
    
    # Export results
    export_csv(results, args.output_csv)
    export_latex(results)


if __name__ == '__main__':
    main()
