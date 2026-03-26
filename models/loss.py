#Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/models/generator.py
import torch
import torch.nn as nn
import numpy as np
from pesq import pesq
from pystoi import stoi
from joblib import Parallel, delayed

def phase_losses(phase_r, phase_g, cfg):
    dim_freq = cfg['stft_cfg']['n_fft'] // 2 + 1
    dim_time = phase_r.size(-1)

    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) -
                 torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) -
                 torch.eye(dim_freq)).to(phase_g.device)

    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) -
                  torch.triu(torch.ones(dim_time, dim_time), diagonal=2) -
                  torch.eye(dim_time)).to(phase_g.device)

    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r - gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r - iaf_g))

    return ip_loss, gd_loss, iaf_loss

def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def compute_stft(y, n_fft, hop_size, win_size, center, compress_factor=1.0):
    eps = torch.finfo(y.dtype).eps
    hann_window = torch.hann_window(win_size).to(y.device)
    
    stft_spec = torch.stft(
        y, 
        n_fft=n_fft, 
        hop_length=hop_size, 
        win_length=win_size, 
        window=hann_window, 
        center=center, 
        pad_mode='reflect', 
        normalized=False, 
        return_complex=True
    )
    
    real_part = stft_spec.real
    imag_part = stft_spec.imag

    mag = torch.sqrt( real_part.pow(2) + imag_part.pow(2) + eps )
    pha = torch.atan2( real_part + eps, imag_part + eps )

    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    
    return mag, pha, com

def pesq_score(utts_r, utts_g, cfg):
    def eval_pesq(clean_utt, esti_utt, sr):
        try:
            pesq_score = pesq(sr, clean_utt, esti_utt)
        except Exception:
            pesq_score = -1
        return pesq_score

    pesq_scores = Parallel(n_jobs=30)(delayed(eval_pesq)(
        utts_r[i].squeeze().cpu().numpy(),
        utts_g[i].squeeze().cpu().numpy(),
        cfg['stft_cfg']['sampling_rate']
    ) for i in range(len(utts_r)))

    return np.mean(pesq_scores)


def si_sdr_loss(reference, estimation):
    min_len = min(reference.shape[-1], estimation.shape[-1])
    reference = reference[..., :min_len]
    estimation = estimation[..., :min_len]
    reference = reference - reference.mean(dim=-1, keepdim=True)
    estimation = estimation - estimation.mean(dim=-1, keepdim=True)

    dot = torch.sum(reference * estimation, dim=-1, keepdim=True)
    s_ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True) + 1e-8
    proj = dot * reference / s_ref_energy
    noise = estimation - proj
    si_sdr = 10 * torch.log10(
        torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + 1e-8) + 1e-8
    )
    return -si_sdr.mean()


def si_sdr_score(utts_r, utts_g):
    scores = []
    for ref, est in zip(utts_r, utts_g):
        ref = ref.squeeze().cpu().numpy().astype(np.float64)
        est = est.squeeze().cpu().numpy().astype(np.float64)
        min_len = min(len(ref), len(est))
        ref, est = ref[:min_len], est[:min_len]

        ref = ref - np.mean(ref)
        est = est - np.mean(est)

        dot = np.sum(ref * est)
        s_ref_energy = np.sum(ref ** 2) + 1e-8
        proj = dot * ref / s_ref_energy

        noise = est - proj
        val = 10 * np.log10(np.sum(proj ** 2) / (np.sum(noise ** 2) + 1e-8) + 1e-8)
        scores.append(val)

    return np.mean(scores)


def stoi_score(utts_r, utts_g, cfg):
    sr = cfg['stft_cfg']['sampling_rate']

    def eval_stoi(clean_utt, esti_utt, sr):
        try:
            min_len = min(len(clean_utt), len(esti_utt))
            return stoi(clean_utt[:min_len], esti_utt[:min_len], sr, extended=False)
        except Exception:
            return -1

    stoi_scores = Parallel(n_jobs=30)(delayed(eval_stoi)(
        utts_r[i].squeeze().cpu().numpy(),
        utts_g[i].squeeze().cpu().numpy(),
        sr
    ) for i in range(len(utts_r)))

    return np.mean(stoi_scores)

