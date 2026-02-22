# Ronny and Shunjie works
# phase_losses and pesq_score are done. Need 3 more functions at the end.
# Ronny: si_sdr_loss for training (returns negative so optimizer minimizes it)
# and si_sdr_score for eval (same idea as pesq_score but no cfg needed).
# Look up SI-SDR (Le Roux et al. 2019).
# Shunjie: stoi_score using pystoi, same pattern as pesq_score.
# Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/models/generator.py

import torch
import torch.nn as nn
import numpy as np
from pesq import pesq
from joblib import Parallel, delayed

def phase_losses(phase_r, phase_g, cfg):
    """
    Calculate phase losses including in-phase loss, gradient delay loss,
    and integrated absolute frequency loss between reference and generated phases.

    Args:
        phase_r (torch.Tensor): Reference phase tensor of shape (batch, freq, time).
        phase_g (torch.Tensor): Generated phase tensor of shape (batch, freq, time).
        cfg (dict): Config dict with stft_cfg.n_fft etc.

    Returns:
        tuple: Tuple containing in-phase loss, gradient delay loss, and integrated absolute frequency loss.
    """
    dim_freq = cfg['stft_cfg']['n_fft'] // 2 + 1  # Calculate frequency dimension
    dim_time = phase_r.size(-1)  # Calculate time dimension

    # Construct gradient delay matrix
    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) -
                 torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) -
                 torch.eye(dim_freq)).to(phase_g.device)

    # Apply gradient delay matrix to reference and generated phases
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    # Construct integrated absolute frequency matrix
    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) -
                  torch.triu(torch.ones(dim_time, dim_time), diagonal=2) -
                  torch.eye(dim_time)).to(phase_g.device)

    # Apply integrated absolute frequency matrix to reference and generated phases
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    # Calculate losses
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r - gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r - iaf_g))

    return ip_loss, gd_loss, iaf_loss

def anti_wrapping_function(x):
    """
    Anti-wrapping function to adjust phase values within the range of -pi to pi.

    Args:
        x (torch.Tensor): Input tensor representing phase differences.

    Returns:
        torch.Tensor: Adjusted tensor with phase values wrapped within -pi to pi.
    """
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def pesq_score(utts_r, utts_g, cfg):
    """
    Calculate PESQ (Perceptual Evaluation of Speech Quality) score for pairs of reference and generated utterances.

    Args:
        utts_r (list of torch.Tensor): List of reference utterances.
        utts_g (list of torch.Tensor): List of generated utterances.
        cfg (dict): Config dict with stft_cfg.sampling_rate.

    Returns:
        float: Mean PESQ score across all pairs of utterances.
    """
    def eval_pesq(clean_utt, esti_utt, sr):
        """
        Evaluate PESQ score for a single pair of clean and estimated utterances.

        Args:
            clean_utt (np.ndarray): Clean reference utterance.
            esti_utt (np.ndarray): Estimated generated utterance.
            sr (int): Sampling rate.

        Returns:
            float: PESQ score or -1 in case of an error.
        """
        try:
            pesq_score = pesq(sr, clean_utt, esti_utt)
        except Exception as e:
            # Error can happen due to silent period or other issues
            print(f"Error computing PESQ score: {e}")
            pesq_score = -1
        return pesq_score

    # Parallel processing of PESQ score computation
    pesq_scores = Parallel(n_jobs=30)(delayed(eval_pesq)(
        utts_r[i].squeeze().cpu().numpy(),
        utts_g[i].squeeze().cpu().numpy(),
        cfg['stft_cfg']['sampling_rate']
    ) for i in range(len(utts_r)))

    # Calculate mean PESQ score
    pesq_score = np.mean(pesq_scores)
    return pesq_score


# Ronny ---------------------------------------------------------------
def si_sdr_loss(reference, estimation):
    """
    Training loss: returns NEGATIVE SI-SDR so optimizer minimizes it.

    Args:
        reference:  [B, T] clean waveform
        estimation: [B, T] enhanced waveform
    Returns:
        scalar tensor (negative SI-SDR, averaged over batch)
    """
    # TODO look up SI-SDR formula, return negative mean so optimizer can minimize
    # keep it in torch, add eps for log
    raise NotImplementedError


def si_sdr_score(utts_r, utts_g):
    """
    Evaluation metric: mean SI-SDR in dB.
    Same structure as pesq_score but does not need cfg (no sample rate required).

    Args:
        utts_r: list of tensors, reference utterances
        utts_g: list of tensors, enhanced utterances
    Returns:
        float
    """
    # TODO
    raise NotImplementedError


# Shunjie --------------------------------------------------------------
def stoi_score(utts_r, utts_g, cfg):
    """
    STOI evaluation metric using pystoi (pip install pystoi).
    Same structure as pesq_score: parallel, handle exceptions, return mean.

    Args:
        utts_r: list of tensors
        utts_g: list of tensors
        cfg:    config dict (need cfg['stft_cfg']['sampling_rate'])
    Returns:
        float
    """
    from pystoi import stoi
    
    def eval_stoi(clean_utt, esti_utt, sr):
        """
        Evaluate STOI score for a single pair of clean and estimated utterances.

        Args:
            clean_utt (np.ndarray): Clean reference utterance.
            esti_utt (np.ndarray): Estimated generated utterance.
            sr (int): Sampling rate.

        Returns:
            float: STOI score or -1 in case of an error.
        """
        try:
            stoi_score = stoi(clean_utt, esti_utt, sr, extended=False)
        except Exception as e:
            # Error can happen due to silent period or other issues
            print(f"Error computing STOI score: {e}")
            stoi_score = -1
        return stoi_score

    # Parallel processing of STOI score computation
    stoi_scores = Parallel(n_jobs=30)(delayed(eval_stoi)(
        utts_r[i].squeeze().cpu().numpy(),
        utts_g[i].squeeze().cpu().numpy(),
        cfg['stft_cfg']['sampling_rate']
    ) for i in range(len(utts_r)))

    # Calculate mean STOI score
    mean_stoi = np.mean(stoi_scores)
    return mean_stoi
