#Author: Fan
#general purpose AV dataloader, works for any dataset that has audio+video pairs
#if you need dataset specific stuff check dataloader_grid/lrs/vox

import random
import soundfile as sf
import torch
import torch.nn.functional as F
from dataloaders.av_utils import (
    load_json, load_video_frames, mix_audio, apply_visual_aug,
    pad_or_trim_video, normalize_and_stft,
)
from dataloaders.augment_rir import RIRAugmentor

class AVDataset(torch.utils.data.Dataset):
    def __init__(self, data_json, noise_json, cfg, split=True, visual_augmentation=False,
                 rir_json=None, rir_prob=0.3):
        self.data_list = load_json(data_json) if isinstance(data_json, str) else data_json
        self.noise_paths = load_json(noise_json) if isinstance(noise_json, str) else (noise_json or [])
        self.noise_cache = {}
        self.cfg = cfg
        self.split = split
        self.visual_augmentation = visual_augmentation
        self.n_fft = cfg['stft_cfg']['n_fft']
        self.hop_size = cfg['stft_cfg']['hop_size']
        self.win_size = cfg['stft_cfg']['win_size']
        self.compress_factor = cfg['model_cfg']['compress_factor']
        self.sr = cfg['stft_cfg'].get('sampling_rate', 16000)
        self.face_size = cfg['training_cfg'].get('face_size', 96)
        self.fps = cfg['training_cfg'].get('video_fps', 25)
        self.rir = RIRAugmentor(rir_json, rir_prob, self.sr) if rir_json else None

    def _get_noise(self):
        #grab a random noise clip, cache it so we don't reread
        path = random.choice(self.noise_paths)
        if path not in self.noise_cache:
            data, _ = sf.read(path)
            if data.ndim > 1:
                data = data[:, 0]
            self.noise_cache[path] = torch.from_numpy(data).float()
        return self.noise_cache[path]

    def __getitem__(self, index):
        #if a file is broken just try another one, give up after 3 tries
        last_error = None
        for attempt in range(4):
            try:
                idx = random.randint(0, len(self.data_list) - 1) if attempt > 0 else index
                return self._load_sample(idx)
            except Exception as e:
                last_error = e
                if attempt >= 3:
                    raise last_error
        raise last_error

    def _load_sample(self, index):
        sample = self.data_list[index]
        audio_path = sample['audio']
        video_path = sample['video']
        clean_audio, sr = sf.read(audio_path)
        if clean_audio.ndim > 1:
            clean_audio = clean_audio[:, 0]
        clean = torch.from_numpy(clean_audio).float()

        segment_size = self.cfg['training_cfg']['segment_size']

        if self.split:
            if clean.size(0) >= segment_size:
                s = random.randint(0, clean.size(0) - segment_size)
                clean = clean[s:s + segment_size]
                start_sec = s / sr
            else:
                clean = F.pad(clean, (0, segment_size - clean.size(0)))
                start_sec = 0.0
            dur_sec = segment_size / sr
        else:
            start_sec, dur_sec = None, None

        video = load_video_frames(video_path, start_sec, dur_sec, self.face_size, self.fps)
        if self.split:
            target_frames = max(1, int(dur_sec * self.fps))
            video = pad_or_trim_video(video, target_frames)

        if self.visual_augmentation:
            video = apply_visual_aug(video)

        if self.rir:
            clean = self.rir(clean)

        if self.noise_paths:
            snr_db = random.uniform(self.cfg['training_cfg']['snr_range'][0],
                                    self.cfg['training_cfg']['snr_range'][1])
            noisy = mix_audio(clean, self._get_noise(), snr_db)
        else:
            noisy = clean.clone()

        c_aud, c_mag, c_pha, c_com, n_mag, n_pha = normalize_and_stft(
            clean, noisy, self.n_fft, self.hop_size, self.win_size, self.compress_factor)

        return (c_aud, c_mag, c_pha, c_com, n_mag, n_pha, video)

    def __len__(self):
        return len(self.data_list)
