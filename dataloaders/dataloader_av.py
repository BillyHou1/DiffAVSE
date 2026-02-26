# Zhenning (video loading) + Fan (audio pipeline + noise mixing)
# Main dataloader, loads paired audio+video, crops to the same time segment,
# mixes in noise at a random SNR, runs STFT, returns everything for training.
#
# Audio pipeline (Fan) + video loading placeholder (Zhenning)
#
# 
#
# load_video_frames(video_path, start_sec, duration_sec, face_size=96, fps=25)
# read frames from .mpg/.mp4 with cv2.VideoCapture, seek to start_sec * fps,
# each frame: BGR->RGB, center crop to square since GRID is 360x288, resize to
# face_size. If a read fails use a black frame. Stack into tensor, permute to
# [3, Tv, H, W], divide by 255. Don't forget cap.release().
#
# 
#
# apply_visual_augmentation(video_frames)
# random degradation so VCE learns to spot bad frames. Roughly 60% original,
# 8% all-black, 10% random frame dropout, 10% gaussian noise, 7% blur,
# 5% dim brightness. Doesn't need to be exact.
#

#
# 
import numpy as np
import random
import json
import soundfile as sf
import torch
import torch.nn.functional as F
from models.stfts import mag_phase_stft

def load_video_frames(video_path, start_sec, duration_sec, face_size=96, fps=25):
    """Placeholder: video """
    raise NotImplementedError("load_video_frames is not implemented")


def load_json_file(path):
    """
    load_json_file(path), just a json.load wrapper, returns the list
    """
    with open(path, 'r') as f:
        return json.load(f)

def mix_audio(clean, noise, snr_db):
    """
    if noise shorter than clean loop it, if longer random crop. Scale factor:
    scale = sqrt(clean_power / (noise_power * 10^(snr_db/10))), return clean + scale * noise
    """
    if len(clean) == 0:
        return np.array([], dtype=clean.dtype)
    if len(noise) < len(clean):
        noise = np.tile(noise, len(clean) // len(noise) + 1)
    start = random.randint(0, len(noise) - len(clean))
    noise = noise[start:start + len(clean)]
    eps = 1e-8
    clean_power = np.sum(clean**2) + eps
    noise_power = np.sum(noise**2) + eps
    scale = np.sqrt(clean_power / (noise_power * 10**(snr_db/10)))
    return clean + scale * noise


class AVDataset(torch.utils.data.Dataset):
    def __init__(self, data_json, noise_json, cfg, split=True, visual_augmentation=False, rir_augmentor=None):
        """
        AVDataset.__init__ takes data_json, noise_json, cfg, split=True,
        visual_augmentation=False, rir_augmentor=None. Load entries from data_json
        which is a list of {"audio": path, "video": path} dicts, load noise paths
        from noise_json, store config values.  
        Args:
            data_json: list of {"audio": path, "video": path} dicts
            noise_json: list of noise paths
            cfg: config values
        """
        self.data_json = data_json
        self.noise_json = noise_json
        self.cfg = cfg
        self.split = split
        self.visual_augmentation = visual_augmentation
        self.rir_augmentor = rir_augmentor
        self.n_fft = cfg['stft_cfg']['n_fft']
        self.hop_size = cfg['stft_cfg']['hop_size']
        self.win_size = cfg['stft_cfg']['win_size']
        self.compress_factor = cfg['model_cfg']['compress_factor']

    def __getitem__(self, index):
        """
        AVDataset.__getitem__ returns a 7-tuple. Load clean audio with soundfile,
        if stereo take ch0, convert to tensor. If split=True random crop to
        segment_size, if shorter pad zeros. Figure out start_sec and duration_sec
        for the matching video crop. Load video aligned to audio, if split=False
        load full audio. Apply visual augmentation if enabled. Apply RIR if
        augmentor is set. Pick random noise + random SNR, mix_audio. RMS normalize
        with norm_factor = sqrt(N / sum(noisy^2)), apply SAME factor to both clean
        and noisy or the loss breaks. STFT both, return clean_audio, clean_mag,
        clean_pha, clean_com, noisy_mag, noisy_pha, video_frames.
        If getitem crashes on a bad file catch it and retry random index, up to 3x.
        """
        last_error = None
        for attempt in range(4): 
            try:
                idx = random.randint(0, len(self.data_json) - 1) if attempt > 0 else index
                return self._load_sample(idx)
            except Exception as e:
                last_error = e
                if attempt >= 3:
                    raise last_error
        raise last_error

    def _load_sample(self, index):
        """Load a single sample by index. Used by __getitem__ with retry logic."""
        sample = self.data_json[index]
        audio_path = sample['audio']
        video_path = sample['video']
        clean_audio, sr = sf.read(audio_path)
        #if stereo take ch0, convert to tensor.
        if clean_audio.ndim> 1:   
            clean_audio = clean_audio[:, 0]
        #if split=True random crop to segment_size, if shorter pad zeros.
        segment_size = self.cfg['training_cfg']['segment_size']
        audio_start = 0
        video_sec_start = 0.0
        video_sec_duration = len(clean_audio) / sr
        if self.split:
            clean_len = len(clean_audio)
            if clean_len >= segment_size:
                max_audio_start = clean_len - segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[audio_start:audio_start+segment_size]
            else:
                clean_audio = np.pad(clean_audio, (0, segment_size - clean_len), 'constant')
        #figure out start_sec and duration_sec for the matching video crop.
            video_sec_start = audio_start / sr
            video_sec_duration = segment_size / sr 
            # TODO: add video load based on video_sec_start and video_sec_duration
        else:
            pass
            # TODO: if split=False load full audio,add video load here
            
        # TODO: apply visual augmentation if enabled

        # apply RIR if augmentor is set
        if self.rir_augmentor:
            clean_audio = self.rir_augmentor(clean_audio)
        snr_db = random.uniform(self.cfg['training_cfg']['snr_range'][0], self.cfg['training_cfg']['snr_range'][1])
        # pick random noise + random SNR, mix_audio
        noise_path = random.choice(self.noise_json)
        noise_audio, _ = sf.read(noise_path)
        if noise_audio.ndim > 1:
            noise_audio = noise_audio[:, 0]
        noisy_audio = mix_audio(clean_audio, noise_audio, snr_db)
        # RMS normalize with norm_factor = sqrt(N / sum(noisy^2)), apply SAME factor to both clean and noisy or the loss breaks.
        norm_factor = np.sqrt(len(noisy_audio) / (np.sum(noisy_audio**2) + 1e-8))
        clean_audio = clean_audio * norm_factor
        noisy_audio = noisy_audio * norm_factor
        # 在这里打包成TENSOR
        clean_audio = torch.FloatTensor(clean_audio) 
        noisy_audio = torch.FloatTensor(noisy_audio)
        # STFT both, return clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha, video_frames.
        clean_mag, clean_pha, clean_com = mag_phase_stft(clean_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)
        noisy_mag, noisy_pha, noisy_com = mag_phase_stft(noisy_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)
        # TODO: video_frames from load_video_frames(video_path, ...) when video loading is implemented
        video_frames = None
        return (clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha, video_frames)
    
    def __len__(self):
        return len(self.data_json)