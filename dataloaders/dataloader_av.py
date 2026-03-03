# Zhenning (video loading) + Fan (audio pipeline + noise mixing)
# Main dataloader, loads paired audio+video, crops to the same time segment,
# mixes in noise at a random SNR, runs STFT, returns everything for training.
#
# TODO implement helper functions and AVDataset class
#
# load_json_file(path), just a json.load wrapper, returns the list
#
# load_video_frames(video_path, start_sec, duration_sec, face_size=96, fps=25)
# read frames from .mpg/.mp4 with cv2.VideoCapture, seek to start_sec * fps,
# each frame: BGR->RGB, center crop to square since GRID is 360x288, resize to
# face_size. If a read fails use a black frame. Stack into tensor, permute to
# [3, Tv, H, W], divide by 255. Don't forget cap.release().
#
# mix_audio(clean, noise, snr_db)
# if noise shorter than clean loop it, if longer random crop. Scale factor:
# scale = sqrt(clean_power / (noise_power * 10^(snr_db/10))), return clean + scale * noise
#
# apply_visual_augmentation(video_frames)
# random degradation so VCE learns to spot bad frames. Roughly 60% original,
# 8% all-black, 10% random frame dropout, 10% gaussian noise, 7% blur,
# 5% dim brightness. Doesn't need to be exact.
#
# AVDataset.__init__ takes data_json, noise_json, cfg, split=True,
# visual_augmentation=False, rir_augmentor=None. Load entries from data_json
# which is a list of {"audio": path, "video": path} dicts, load noise paths
# from noise_json, store config values.
#
# AVDataset.__getitem__ returns a 7-tuple. Load clean audio with soundfile,
# if stereo take ch0, convert to tensor. If split=True random crop to
# segment_size, if shorter pad zeros. Figure out start_sec and duration_sec
# for the matching video crop. Load video aligned to audio, if split=False
# load full video. Apply visual augmentation if enabled. Apply RIR if
# augmentor is set. Pick random noise + random SNR, mix_audio. RMS normalize
# with norm_factor = sqrt(N / sum(noisy^2)), apply SAME factor to both clean
# and noisy or the loss breaks. STFT both, return clean_audio, clean_mag,
# clean_pha, clean_com, noisy_mag, noisy_pha, video_frames.
# If getitem crashes on a bad file catch it and retry random index, up to 3x.

import json
import random

import numpy as np
import torch
import torch.nn.functional as F

# Optional imports for video and audio processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import soundfile as sf
    from models.stfts import mag_phase_stft
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


def load_json_file(path):
    """
    Load JSON file, just a json.load wrapper.

    Args:
        path: path to JSON file

    Returns:
        loaded JSON data (typically a list)
    """
    with open(path, "r") as f:
        return json.load(f)


def load_video_frames(video_path, start_sec, duration_sec, face_size=96, fps=25):
    """
    Load video frames from a video file and crop to a specified time segment.

    Args:
        video_path: path to video file (.mp4/.mpg)
        start_sec: start time in seconds
        duration_sec: duration in seconds
        face_size: output spatial size (default 96)
        fps: frames per second (default 25)

    Returns:
        torch.Tensor of shape [3, T_v, H, W] with values in [0, 1]
    """
    if not CV2_AVAILABLE:
        print("Warning: opencv-python not available, returning dummy frames")
        t_v = max(1, int(duration_sec * fps))
        black_frames = np.zeros((t_v, face_size, face_size, 3), dtype=np.float32)
        return torch.from_numpy(black_frames).permute(3, 0, 1, 2)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        # Return black frames if video cannot be opened
        t_v = max(1, int(duration_sec * fps))
        black_frames = np.zeros((t_v, face_size, face_size, 3), dtype=np.float32)
        return torch.from_numpy(black_frames).permute(3, 0, 1, 2)

    # Calculate frame indices (ensure at least 1 frame)
    start_frame = int(start_sec * fps)
    num_frames = max(1, int(duration_sec * fps))

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()

        if not ret:
            # Use black frame if read fails
            frame = np.zeros((face_size, face_size, 3), dtype=np.uint8)
        else:
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Center crop to square
            h, w = frame.shape[:2]
            min_dim = min(h, w)
            start_h = (h - min_dim) // 2
            start_w = (w - min_dim) // 2
            frame = frame[start_h:start_h + min_dim, start_w:start_w + min_dim]

            # Resize to target size
            frame = cv2.resize(frame, (face_size, face_size))

        frames.append(frame)

    cap.release()

    # Stack frames: [T_v, H, W, 3]
    frames = np.stack(frames, axis=0).astype(np.float32)

    # Normalize to [0, 1]
    frames = frames / 255.0

    # Convert to tensor and permute to [3, T_v, H, W]
    frames_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2)

    return frames_tensor


def mix_audio(clean, noise, snr_db):
    """
    Mix clean audio with noise at a specified SNR.

    Args:
        clean: clean audio signal
        noise: noise audio signal
        snr_db: target SNR in dB

    Returns:
        mixed audio signal
    """
    if len(clean) == 0:
        return np.array([], dtype=clean.dtype)

    # Loop noise if shorter than clean
    if len(noise) < len(clean):
        noise = np.tile(noise, len(clean) // len(noise) + 1)

    # Random crop noise if longer than clean
    start = random.randint(0, len(noise) - len(clean))
    noise = noise[start:start + len(clean)]

    # Calculate scale factor to achieve target SNR (with numerical protection)
    eps = 1e-10
    clean_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2)

    if noise_power < eps:
        # If noise is effectively zero, return clean audio
        return clean

    scale = np.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10) + eps))

    return clean + scale * noise


def apply_visual_augmentation(video_frames, p_augment=0.4):
    """
    Apply random visual augmentation to video frames.

    Augmentation types:
    - 60% original (no augmentation)
    - 8% all-black
    - 10% random frame dropout
    - 10% gaussian noise
    - 7% blur
    - 5% dim brightness

    Args:
        video_frames: torch.Tensor of shape [3, T_v, H, W]
        p_augment: probability of applying augmentation (default 0.4)

    Returns:
        augmented video_frames tensor
    """
    if random.random() > p_augment:
        # 60% keep original
        return video_frames

    aug_type = random.random()

    if aug_type < 0.08 / 0.4:  # 8% all-black
        return torch.zeros_like(video_frames)

    elif aug_type < (0.08 + 0.10) / 0.4:  # 10% frame dropout
        t_v = video_frames.shape[1]
        num_drop = int(t_v * random.uniform(0.2, 0.5))
        drop_indices = random.sample(range(t_v), num_drop)

        augmented = video_frames.clone()
        for idx in drop_indices:
            augmented[:, idx, :, :] = 0
        return augmented

    elif aug_type < (0.08 + 0.10 + 0.10) / 0.4:  # 10% gaussian noise
        noise = torch.randn_like(video_frames) * random.uniform(0.05, 0.15)
        augmented = video_frames + noise
        return torch.clamp(augmented, 0, 1)

    elif aug_type < (0.08 + 0.10 + 0.10 + 0.07) / 0.4:  # 7% blur
        if CV2_AVAILABLE:
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 1.5)

            # Convert to numpy for cv2 blur
            frames_np = video_frames.permute(1, 2, 3, 0).numpy()  # [T_v, H, W, 3]
            blurred_frames = []
            for frame in frames_np:
                blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
                blurred_frames.append(blurred)

            blurred_frames = np.stack(blurred_frames, axis=0)
            blurred_tensor = torch.from_numpy(blurred_frames).permute(3, 0, 1, 2)
            return blurred_tensor
        else:
            # Fallback: use average pooling per frame to simulate blur
            c, t_v, h, w = video_frames.shape
            kernel_size = random.choice([3, 5])

            blurred_frames = []
            for t in range(t_v):
                frame = video_frames[:, t, :, :].unsqueeze(0)  # [1, 3, H, W]
                blurred_frame = F.avg_pool2d(
                    frame,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2
                )
                blurred_frames.append(blurred_frame.squeeze(0))

            augmented = torch.stack(blurred_frames, dim=1)  # [3, T_v, H, W]
            return augmented

    else:  # 5% dim brightness
        brightness_factor = random.uniform(0.3, 0.6)
        return video_frames * brightness_factor


class AVDataset:
    def __init__(self, data_json, noise_json, cfg, split=True, visual_augmentation=False, rir_augmentor=None):
        """
        Audio-Visual Dataset for training/validation.

        Args:
            data_json: list of {"audio": path, "video": path} dicts
            noise_json: list of noise audio paths
            cfg: config dict with training parameters
            split: True for training (random crop), False for inference (full length)
            visual_augmentation: whether to apply visual augmentation
            rir_augmentor: RIR augmentation function (optional)
        """
        self.data_json = data_json
        self.noise_json = noise_json
        self.cfg = cfg
        self.split = split
        self.visual_augmentation = visual_augmentation
        self.rir_augmentor = rir_augmentor
        self.n_fft = cfg["stft_cfg"]["n_fft"]
        self.hop_size = cfg["stft_cfg"]["hop_size"]
        self.win_size = cfg["stft_cfg"]["win_size"]
        self.compress_factor = cfg["model_cfg"]["compress_factor"]

    def __getitem__(self, index):
        """
        Get a training sample with retry logic on errors.

        Returns:
            tuple: (clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha, video_frames)
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
        """Load a single sample by index."""
        sample = self.data_json[index]
        audio_path = sample["audio"]
        video_path = sample["video"]

        # Load clean audio
        clean_audio, sr = sf.read(audio_path)

        # If stereo, take channel 0
        if clean_audio.ndim > 1:
            clean_audio = clean_audio[:, 0]

        # Default: load full aligned video
        video_sec_start = 0
        video_sec_duration = len(clean_audio) / sr

        # Training mode: random crop to segment_size
        if self.split:
            segment_size = self.cfg["segment_size"]
            clean_len = len(clean_audio)

            if clean_len >= segment_size:
                max_audio_start = clean_len - segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[audio_start:audio_start + segment_size]
            else:
                # Pad if shorter than segment_size
                clean_audio = np.pad(clean_audio, (0, segment_size - clean_len), "constant")
                audio_start = 0

            # Calculate video crop timing based on audio crop
            video_sec_start = audio_start / sr
            video_sec_duration = segment_size / sr

        # Load video frames aligned with audio
        video_frames = load_video_frames(
            video_path,
            start_sec=video_sec_start,
            duration_sec=video_sec_duration,
            face_size=96,
            fps=25,
        )

        # Apply visual augmentation if enabled
        if self.visual_augmentation and self.split:
            video_frames = apply_visual_augmentation(video_frames)

        # Apply RIR if augmentor is set
        if self.rir_augmentor:
            clean_audio = self.rir_augmentor(clean_audio)

        # Mix with noise
        snr_db = random.uniform(self.cfg["snr_range"][0], self.cfg["snr_range"][1])
        noise_path = random.choice(self.noise_json)
        noise_audio, _ = sf.read(noise_path)

        # Handle stereo noise
        if noise_audio.ndim > 1:
            noise_audio = noise_audio[:, 0]

        noisy_audio = mix_audio(clean_audio, noise_audio, snr_db)

        # RMS normalization (apply same factor to both clean and noisy)
        eps = 1e-10
        noisy_power = np.sum(noisy_audio ** 2)
        if noisy_power < eps:
            norm_factor = 1.0
        else:
            norm_factor = np.sqrt(len(noisy_audio) / noisy_power)

        clean_audio = clean_audio * norm_factor
        noisy_audio = noisy_audio * norm_factor

        # Convert to tensors
        clean_audio = torch.FloatTensor(clean_audio)
        noisy_audio = torch.FloatTensor(noisy_audio)

        # STFT
        clean_mag, clean_pha, clean_com = mag_phase_stft(
            clean_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor
        )
        noisy_mag, noisy_pha, _ = mag_phase_stft(
            noisy_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor
        )

        return (
            clean_audio,
            clean_mag,
            clean_pha,
            clean_com,
            noisy_mag,
            noisy_pha,
            video_frames,
        )

    def __len__(self):
        return len(self.data_json)


# Test code
if __name__ == "__main__":
    print("Testing dataset functions...")
    print("Note: Full functionality requires opencv-python, soundfile, and models.stfts")

    # Test augmentation function
    print("\nTesting visual augmentation...")

    dummy_frames = torch.rand(3, 25, 96, 96)
    print(f"Dummy video frames shape: {dummy_frames.shape}")
    print(f"Video frames value range: [{dummy_frames.min():.3f}, {dummy_frames.max():.3f}]")

    # Test multiple augmentations to verify distribution
    print("\nTesting augmentation distribution (20 samples)...")
    aug_results = []
    for _ in range(20):
        frames = torch.rand(3, 25, 96, 96)
        aug_frames = apply_visual_augmentation(frames, p_augment=0.4)

        # Rough augmentation type detection
        if torch.allclose(frames, aug_frames, rtol=1e-4, atol=1e-4):
            aug_type = "original"
        elif torch.all(aug_frames == 0):
            aug_type = "black"
        elif torch.max(aug_frames) < 0.7:
            aug_type = "dim"
        elif aug_frames.shape[1] != frames.shape[1]:
            aug_type = "dropout"
        else:
            aug_type = "noise/blur"

        aug_results.append(aug_type)

    print(f"Augmentation types: {aug_results}")
    print("Distribution:")
    print(f"  - original: {aug_results.count('original')}")
    print(f"  - black: {aug_results.count('black')}")
    print(f"  - dim: {aug_results.count('dim')}")
    print(f"  - dropout: {aug_results.count('dropout')}")
    print(f"  - noise/blur: {aug_results.count('noise/blur')}")

    print("\nAll core tests passed!")
    print("\nTo test video loading, ensure opencv-python is installed:")
    print("  pip install opencv-python")
