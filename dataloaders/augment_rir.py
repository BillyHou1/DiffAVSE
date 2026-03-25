#Author: Fan
#room impulse response stuff-convolve clean audio with RIR to fake room acoustics
import json
import soundfile as sf
import librosa
import random
import scipy
import numpy as np
import torch

def apply_rir(audio, rir):
    is_tensor = isinstance(audio, torch.Tensor)
    audio_np = audio.numpy() if is_tensor else audio
    if rir.ndim > 1:
        rir = rir[:, 0] if rir.shape[1] >= 1 else rir.ravel()
    rir = np.asarray(rir, dtype=np.float64).ravel()
    rst = scipy.signal.fftconvolve(audio_np, rir, mode='full')[:len(audio_np)]
    audio_energy = np.sqrt(np.sum(audio_np**2))
    rst_energy = np.sqrt(np.sum(rst**2))
    if rst_energy <= 0:
        return audio
    rst = rst * (audio_energy / rst_energy)
    if is_tensor:
        return torch.from_numpy(rst.astype(np.float32))
    return rst

class RIRAugmentor:
    def __init__(self, rir_json, prob=0.3, target_sr=16000):
        with open(rir_json, 'r') as f:
            self.rir_list = json.load(f)
        self.prob = prob
        self.target_sr = target_sr
        self.cache = {}

    def _load_rir(self, rir_path):
        #only load when we actually need it, then keep it around
        if rir_path in self.cache:
            return self.cache[rir_path]
        rir_data, sr = sf.read(rir_path)
        if rir_data.ndim > 1:
            rir_data = rir_data[:, 0]
        if sr != self.target_sr:
            rir_data = librosa.resample(rir_data, orig_sr=sr, target_sr=self.target_sr)
        #normalize so different RIR files have similar energy
        rir_data = rir_data / (np.sqrt(np.sum(rir_data**2)) + 1e-10)
        self.cache[rir_path] = rir_data
        return rir_data

    def __call__(self, audio):
        if random.random() >= self.prob or len(self.rir_list) == 0:
            return audio
        rir_path = random.choice(self.rir_list)
        rir = self._load_rir(rir_path)
        return apply_rir(audio, rir)