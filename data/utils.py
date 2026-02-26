import os
import json
import subprocess

def save_json(data, path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def extract_audio_from_video(video_path, sr=16000):
    """
    extract_audio_from_video(video_path), extract audio from video using ffmpeg
    """
    audio_path = os.path.join(os.path.dirname(video_path), os.path.splitext(os.path.basename(video_path))[0] + '.wav')
    ret = subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-ac', '1', '-ar', str(sr), audio_path],
                         capture_output=True)
    if ret.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {video_path}, stderr: {ret.stderr.decode(errors='replace')}")
    return audio_path