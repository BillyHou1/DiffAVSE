#Author: Fan
#VoxCeleb2 loader, face videos should already be cropped
import random
import torch
import torch.utils.data
from dataloaders.av_utils import (
    load_json, load_video_frames, extract_audio_from_video,
    mix_audio, normalize_and_stft,
)
from dataloaders.augment_rir import RIRAugmentor

class VoxCelebAVDataset(torch.utils.data.Dataset):
    def __init__(self, data_json, noise_json=None, sampling_rate=16000,
                 segment_size=32000, n_fft=400, hop_size=100, win_size=400,
                 compress_factor=1.0, snr_range=(-5, 20), face_size=96,
                 video_fps=25, split=True, shuffle=True,
                 rir_json=None, rir_prob=0.3, min_audio_len=8000):

        self.video_paths = load_json(data_json)
        self.noise_paths = load_json(noise_json) if noise_json else []
        random.seed(1234)
        if shuffle:
            random.shuffle(self.video_paths)
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
        self.min_len = min_audio_len
        self.rir = RIRAugmentor(rir_json, rir_prob, sampling_rate) if rir_json else None

    def _get_noise(self):
        #read a noise file with torchaudio,works for wav and flac
        import torchaudio
        path = random.choice(self.noise_paths)
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0)

    def __getitem__(self, index):
        for attempt in range(3):
            idx = index if attempt == 0 else random.randint(0, len(self) - 1)
            try:
                return self._load(idx)
            except Exception as e:
                if attempt == 0:
                    print(f"Vox skip {index}: {e}")
        raise RuntimeError(f"VoxCeleb2 failed after 3 attempts")

    def _load(self, index):
        vpath = self.video_paths[index]
        clean = extract_audio_from_video(vpath, self.sr)

        if clean.size(0) < self.min_len:
            clean = torch.nn.functional.pad(clean, (0, self.min_len - clean.size(0)))

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

        video = load_video_frames(vpath, start_sec, dur_sec, self.face_size, self.fps)

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
        return len(self.video_paths)
