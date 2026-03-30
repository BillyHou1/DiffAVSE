#Author: Fan
#GRID dataset loader, reads wav+mpg pairs and mixes noise on the fly

import random
import torch
import torch.utils.data
import soundfile as sf
from dataloaders.av_utils import (
    load_json, load_video_frames, mix_audio, apply_visual_aug,
    pad_or_trim_video, normalize_and_stft,)
from dataloaders.augment_rir import RIRAugmentor

class GRIDAVDataset(torch.utils.data.Dataset):
    def __init__(self, data_json, noise_json=None, sampling_rate=16000,
                 segment_size=16000, n_fft=400, hop_size=100, win_size=400,
                 compress_factor=1.0, snr_range=(-5, 20), face_size=96,
                 video_fps=25, split=True, shuffle=True,
                 visual_augmentation=False, rir_json=None, rir_prob=0.3):

        self.entries = load_json(data_json)
        self.noise_paths = load_json(noise_json) if noise_json else []
        self.noise_cache = {}
        random.seed(1234)
        if shuffle:
            random.shuffle(self.entries)
        self.sr = sampling_rate
        self.seg = segment_size
        self.n_fft = n_fft
        self.hop = hop_size
        self.win = win_size
        self.compress = compress_factor
        self.snr_min, self.snr_max = snr_range
        self.face_size = face_size
        self.fps = video_fps
        self.split = split
        self.vis_aug = visual_augmentation
        self.rir = RIRAugmentor(rir_json, rir_prob, sampling_rate) if rir_json else None

    def _get_noise(self):
        path = random.choice(self.noise_paths)
        if path not in self.noise_cache:
            data, _ = sf.read(path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            self.noise_cache[path] = torch.from_numpy(data).float()
        return self.noise_cache[path]

    def __getitem__(self, index):
        for attempt in range(3):
            idx = index if attempt == 0 else random.randint(0, len(self) - 1)
            try:
                return self._load(idx)
            except Exception as e:
                if attempt == 0:
                    print(f"GRID skip {index}: {e}")
        raise RuntimeError(f"GRID failed after 3 attempts")

    def _load(self, index):
        entry = self.entries[index]
        data, _ = sf.read(entry['audio'])
        clean = torch.from_numpy(data).float()

        if self.split:
            if clean.size(0) >= self.seg:
                s = random.randint(0, clean.size(0) - self.seg)
                clean = clean[s:s + self.seg]
                start_sec = s / self.sr
            else:
                clean = torch.nn.functional.pad(clean, (0, self.seg - clean.size(0)))
                start_sec = 0.0
            dur_sec = self.seg / self.sr
        else:
            start_sec, dur_sec = None, None

        video = load_video_frames(entry['video'], start_sec, dur_sec, self.face_size, self.fps)
        if self.split:
            video = pad_or_trim_video(video, max(1, int(dur_sec * self.fps)))
        if self.vis_aug:
            video = apply_visual_aug(video)
        if self.rir:
            clean = self.rir(clean)
        if self.noise_paths:
            noisy = mix_audio(clean, self._get_noise(), random.uniform(self.snr_min, self.snr_max))
        else:
            noisy = clean.clone()
        c_aud, c_mag, c_pha, c_com, n_mag, n_pha = normalize_and_stft(
            clean, noisy, self.n_fft, self.hop, self.win, self.compress)
        return c_aud, c_mag, c_pha, c_com, n_mag, n_pha, video

    def __len__(self):
        return len(self.entries)
