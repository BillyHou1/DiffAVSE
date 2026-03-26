#Author: Fan
#LRS2 loader-audio gets extracted from the video files
#has extra tricks: visual degradation, wrongface swap, cocktail party noise

import random
import torch
import torch.utils.data
from dataloaders.av_utils import (
    load_json, load_video_frames, extract_audio_from_video,
    mix_audio, apply_visual_degradation, pad_or_trim_video,
    normalize_and_stft,
)
from dataloaders.augment_rir import RIRAugmentor

class LRSAVDataset(torch.utils.data.Dataset):
    def __init__(self, data_json, noise_json=None, sampling_rate=16000,
                 segment_size=32000, n_fft=400, hop_size=100, win_size=400,
                 compress_factor=1.0, snr_range=(-5, 20), face_size=96,
                 video_fps=25, split=True, shuffle=True,
                 rir_json=None, rir_prob=0.3,
                 visual_degradation_prob=0.0,
                 modality_conflict_prob=0.0,
                 cocktail_party_prob=0.0):

        self.entries = load_json(data_json)
        self.noise_paths = load_json(noise_json) if noise_json else []
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
        self.rir = RIRAugmentor(rir_json, rir_prob, sampling_rate) if rir_json else None
        self.vis_deg_prob = visual_degradation_prob
        self.mod_conflict_prob = modality_conflict_prob
        self.cocktail_prob = cocktail_party_prob

    def _get_noise(self):
        #pick a random noise file and read it with pyav
        path = random.choice(self.noise_paths)
        return extract_audio_from_video(path, self.sr)

    def __getitem__(self, index):
        for attempt in range(3):
            idx = index if attempt == 0 else random.randint(0, len(self) - 1)
            try:
                return self._load(idx)
            except Exception as e:
                if attempt == 0:
                    print(f"LRS2 skip {index}: {e}")
        raise RuntimeError(f"LRS2 failed after 3 attempts")

    def _load(self, index):
        entry = self.entries[index]
        vpath = entry['video'] if isinstance(entry, dict) else entry
        clean = extract_audio_from_video(vpath, self.sr)
        vis_degraded = False

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

        #make sure video has the right number of frames
        if self.split:
            expected_v = int(self.seg * self.fps / self.sr)
            tv = video.shape[1]
            if tv < expected_v:
                pad = video[:, -1:, :, :].expand(-1, expected_v - tv, -1, -1)
                video = torch.cat([video, pad], dim=1)
            elif tv > expected_v:
                video = video[:, :expected_v, :, :]

        #randomly mess up the video so the model can't rely on perfect visuals
        if self.vis_deg_prob > 0 and self.split and random.random() <= self.vis_deg_prob:
            video = apply_visual_degradation(video, fps=self.fps, min_dur_ms=200, max_dur_ms=500)
            vis_degraded = True

        #modality conflict: show a different person's face on purpose
        if self.mod_conflict_prob > 0 and self.split and random.random() < self.mod_conflict_prob:
            swap_idx = random.randint(0, len(self.entries) - 1)
            swap_entry = self.entries[swap_idx]
            swap_vpath = swap_entry['video'] if isinstance(swap_entry, dict) else swap_entry
            try:
                video = load_video_frames(swap_vpath, start_sec, dur_sec, self.face_size, self.fps)
                tv = video.shape[1]
                if self.split:
                    if tv < expected_v:
                        pad = video[:, -1:, :, :].expand(-1, expected_v - tv, -1, -1)
                        video = torch.cat([video, pad], dim=1)
                    elif tv > expected_v:
                        video = video[:, :expected_v, :, :]
                vis_degraded = True
            except Exception:
                pass

        if self.rir:
            clean = self.rir(clean)

        #cocktail party: mix in another person talking as interference
        if self.cocktail_prob > 0 and self.split and random.random() < self.cocktail_prob:
            int_idx = random.randint(0, len(self.entries) - 1)
            int_entry = self.entries[int_idx]
            int_vpath = int_entry['video'] if isinstance(int_entry, dict) else int_entry
            try:
                int_audio = extract_audio_from_video(int_vpath, self.sr)
                if int_audio.shape[0] > clean.shape[0]:
                    i_start = random.randint(0, int_audio.shape[0] - clean.shape[0])
                    int_audio = int_audio[i_start:i_start + clean.shape[0]]
                elif int_audio.shape[0] < clean.shape[0]:
                    int_audio = torch.nn.functional.pad(int_audio, (0, clean.shape[0] - int_audio.shape[0]))
                snr_db = random.uniform(self.snr_min, self.snr_max)
                noisy = mix_audio(clean, int_audio, snr_db)
            except Exception:
                if self.noise_paths:
                    noisy = mix_audio(clean, self._get_noise(), random.uniform(self.snr_min, self.snr_max))
                else:
                    noisy = clean.clone()
        elif self.noise_paths:
            noisy = mix_audio(clean, self._get_noise(), random.uniform(self.snr_min, self.snr_max))
        else:
            noisy = clean.clone()

        c_aud, c_mag, c_pha, c_com, n_mag, n_pha = normalize_and_stft(
            clean, noisy, self.n_fft, self.hop, self.win, self.compress)

        return (c_aud, c_mag, c_pha, c_com, n_mag, n_pha, video,
                torch.tensor(vis_degraded, dtype=torch.float32))

    def __len__(self):
        return len(self.entries)
