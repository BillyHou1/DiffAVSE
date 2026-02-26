# Author: Fan
# RIR augmentation: convolve clean speech with room impulse responses.
import json
import soundfile as sf
import librosa
import random
import scipy
import numpy as np

def apply_rir(audio, rir):
    if rir.ndim > 1:
        rir = rir[:, 0] if rir.shape[1] >= 1 else rir.ravel()
    rir = np.asarray(rir, dtype=np.float64).ravel()
    rst = scipy.signal.fftconvolve(audio, rir, mode='full')[:len(audio)]
    audio_energy = np.sqrt(np.sum(audio**2))
    rst_energy = np.sqrt(np.sum(rst**2))
    if rst_energy <= 0:
        return audio
    rst = rst * (audio_energy / rst_energy)
    return rst

class RIRAugmentor:
    def __init__(self, rir_json, prob=0.3, target_sr=16000):
        # load all RIR files into memory at init
        with open(rir_json, 'r') as f:
            self.rir_list = json.load(f)
        self.rirs = []
        for rir in self.rir_list:
            rir_path = rir
            rir_data, sr = sf.read(rir_path)
            if sr != target_sr:
                rir_data = librosa.resample(rir_data, orig_sr=sr, target_sr=target_sr)
            self.rirs.append(rir_data)
        self.prob = prob
        self.target_sr = target_sr

    def __call__(self, audio):
        if random.random() >= self.prob:
            return audio
        rir = random.choice(self.rirs)
        return apply_rir(audio, rir)

   
