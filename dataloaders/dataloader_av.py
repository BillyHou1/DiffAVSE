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
