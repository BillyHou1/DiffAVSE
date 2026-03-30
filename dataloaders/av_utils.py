#Author:Fan
#bunch of helper stuff that all the dataloaders need

import json
import os
import random
import numpy as np
import torch
import cv2
from models.stfts import mag_phase_stft

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_video_frames(video_path, start_sec=None, duration_sec=None, face_size=96, fps=25):
    #read video, center crop the face, resize to face_size output shape [3, T, H, W], values 0 to 1
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    src_fps = float(src_fps) if src_fps and src_fps > 0 else float(fps)
    fps = float(fps) if fps and fps > 0 else 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(round(start_sec * src_fps)) if start_sec is not None else 0
    start_frame = max(0, min(start_frame, max(0, total_frames - 1)))
    target_frames = int(duration_sec * fps) if duration_sec is not None else max(1, int((total_frames - start_frame) * fps / src_fps))
    stride = src_fps / fps
    target_indices = [
        min(total_frames - 1, int(round(start_frame + i * stride)))
        for i in range(target_frames)
    ]

    if total_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    target_ptr = 0
    current_idx = start_frame
    while target_ptr < len(target_indices) and current_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        wanted_idx = target_indices[target_ptr]
        if current_idx != wanted_idx:
            current_idx += 1
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        d = min(h, w)
        frame = frame[(h - d) // 2:(h + d) // 2, (w - d) // 2:(w + d) // 2]
        frame = cv2.resize(frame, (face_size, face_size))
        frames.append(frame)
        target_ptr += 1
        while target_ptr < len(target_indices) and target_indices[target_ptr] == current_idx:
            frames.append(frame.copy())
            target_ptr += 1
        current_idx += 1
    cap.release()

    if len(frames) == 0:
        frames = [np.zeros((face_size, face_size, 3), dtype=np.uint8)]
    return torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 255.0

def extract_audio_from_video(video_path, target_sr=16000):
    #pull audio out of a video file with pyav, gives back a 1d tensor
    import av
    container = av.open(video_path)
    stream = container.streams.audio[0]
    frames = [f.to_ndarray() for f in container.decode(stream)]
    orig_sr = stream.sample_rate
    container.close()
    wav = torch.from_numpy(np.concatenate(frames, axis=1).mean(0).astype("float32"))
    if orig_sr != target_sr:
        import torchaudio
        wav = torchaudio.transforms.Resample(orig_sr, target_sr)(wav.unsqueeze(0)).squeeze(0)
    return wav

def extract_audio_ffmpeg(video_path, sr=16000):
    #same thing but shells out to ffmpeg instead, returns the wav path
    import subprocess
    audio_path = os.path.join(os.path.dirname(video_path),
                              os.path.splitext(os.path.basename(video_path))[0] + '.wav')
    ret = subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-ac', '1', '-ar', str(sr), audio_path],
                         capture_output=True)
    if ret.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {video_path}, stderr: {ret.stderr.decode(errors='replace')}")
    return audio_path

def mix_audio(clean, noise, snr_db):
    #add noise to clean audio at some SNR level
    #if noise is too short it gets looped, too long gets cropped
    n = clean.size(0)
    if noise.size(0) >= n:
        s = random.randint(0, noise.size(0) - n)
        noise = noise[s:s + n]
    else:
        noise = noise.repeat(n // noise.size(0) + 1)[:n]
    clean_power = torch.mean(clean ** 2) + 1e-10
    noise_power = torch.mean(noise ** 2) + 1e-10
    scale = torch.sqrt(clean_power / (10 ** (snr_db / 10.0) * noise_power))
    return clean + scale * noise

def apply_visual_aug(frames):
    #mess up the video randomly so VCE learns to be robust
    #blackout, dropout frames, add noise, blur, or dim it
    r = random.random()
    if r < 0.08:
        return torch.zeros_like(frames)
    elif r < 0.18:
        t_len = frames.size(1)
        mask = torch.rand(t_len) > random.uniform(0.2, 0.8)
        frames[:, ~mask] = 0
        return frames
    elif r < 0.28:
        return torch.clamp(frames + torch.randn_like(frames) * random.uniform(0.1, 0.6), 0, 1)
    elif r < 0.35:
        c, t_len, h, w = frames.shape
        k = random.choice([3, 5, 7])
        return torch.nn.functional.avg_pool2d(
            frames.reshape(c * t_len, 1, h, w), k, stride=1, padding=k // 2
        ).reshape(c, t_len, h, w)
    elif r < 0.40:
        return frames * random.uniform(0.1, 0.5)
    return frames

def apply_visual_degradation(video, fps=25, min_dur_ms=200, max_dur_ms=500):
    #fancier version of visual aug, picks a random time window and applies cosine shaped degradation blur/blackout/freeze/lip cover/lighting
    t_len = video.shape[1]
    min_f = max(3, int(min_dur_ms * fps / 1000))
    max_f = max(min_f + 1, int(max_dur_ms * fps / 1000))
    max_idx = min(max_f, t_len - 2)
    if min_f > max_idx:
        return video
    win_len = random.randint(min_f, max_idx)
    start = random.randint(0, t_len - win_len - 1)
    t = torch.linspace(0, np.pi, win_len)
    curve = 0.5 * (1.0 + torch.cos(t))
    depth = random.uniform(0.3, 1.0)
    curve = 1.0 - depth * (1.0 - curve)
    curve = curve.view(1, win_len, 1, 1)

    deg_type = random.choice(['blur', 'blackout', 'freeze', 'lip_occlude', 'lighting'])
    if deg_type == 'blur':
        degraded = video[:, start:start + win_len].clone()
        for i in range(win_len):
            sigma = (1.0 - curve[0, i, 0, 0].item()) * 8.0 + 0.01
            if sigma > 0.5:
                k = int(sigma * 4) | 1
                k = max(3, min(k, 15))
                for c in range(3):
                    f_np = degraded[c, i].numpy()
                    degraded[c, i] = torch.from_numpy(cv2.GaussianBlur(f_np, (k, k), sigma))
        video[:, start:start + win_len] = degraded
    elif deg_type == 'blackout':
        video[:, start:start + win_len] *= curve
    elif deg_type == 'freeze':
        frozen = video[:, start:start + 1].expand_as(video[:, start:start + win_len])
        video[:, start:start + win_len] = curve * video[:, start:start + win_len] + (1.0 - curve) * frozen
    elif deg_type == 'lip_occlude':
        _, _, h, w = video.shape
        occ_top = int(h * random.uniform(0.45, 0.65))
        fill_val = random.uniform(0.2, 0.6)
        inv_curve = 1.0 - curve
        video[:, start:start + win_len, occ_top:, :] = (
            curve * video[:, start:start + win_len, occ_top:, :] + inv_curve * fill_val
        )
    elif deg_type == 'lighting':
        if random.random() < 0.5:
            dim_floor = random.uniform(0.1, 0.4)
            light_curve = curve * (1.0 - dim_floor) + dim_floor
            video[:, start:start + win_len] *= light_curve
        else:
            bright_ceil = random.uniform(0.7, 0.95)
            inv_curve = 1.0 - curve
            video[:, start:start + win_len] = video[:, start:start + win_len] * curve + bright_ceil * inv_curve
            video.clamp_(0.0, 1.0)
    return video

def pad_or_trim_video(frames, target_t):
    #If too short, repeat the last frame. If too long, chop it off
    t_len = frames.shape[1]
    if t_len < target_t:
        pad = frames[:, -1:].expand(-1, target_t - t_len, -1, -1).clone()
        return torch.cat([frames, pad], dim=1)
    elif t_len > target_t:
        return frames[:, :target_t]
    return frames

def normalize_and_stft(clean, noisy, n_fft, hop_size, win_size, compress_factor):
    #normalize both by noisy rms, then do STFT on both
    clean = clean.unsqueeze(0)
    noisy = noisy.unsqueeze(0)
    norm = torch.sqrt(noisy.size(1) / (torch.sum(noisy ** 2) + 1e-10))
    clean = clean * norm
    noisy = noisy * norm
    c_mag, c_pha, c_com = mag_phase_stft(clean, n_fft, hop_size, win_size, compress_factor)
    n_mag, n_pha, _ = mag_phase_stft(noisy, n_fft, hop_size, win_size, compress_factor)
    return (
        clean.squeeze(), c_mag.squeeze(), c_pha.squeeze(), c_com.squeeze(),
        n_mag.squeeze(), n_pha.squeeze(),
    )
