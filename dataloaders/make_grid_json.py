#Author: Fan
#scan the GRID folder and pair up wav+mpg files by speaker
#s1~s28 for train, s29~s31 valid, s32~s34 test
import os
import argparse
from dataloaders.av_utils import save_json

def get_speaker(parts):
    if len(parts) >= 2 and parts[0] in ("audio", "video"):
        return parts[1]
    return None

def collect_pairs(grid_root):
    grid_root = os.path.abspath(grid_root)
    if not os.path.isdir(grid_root):
        return {}

    audio_paths = {}
    video_paths = {}
    for dirpath, _, filenames in os.walk(grid_root):
        rel = os.path.relpath(dirpath, grid_root)
        parts = rel.split(os.sep)
        spk = get_speaker(parts)
        if spk is None or spk in ("audio", "video"):
            continue
        for f in filenames:
            stem, ext = os.path.splitext(f)
            ext = ext.lower()
            full = os.path.abspath(os.path.join(dirpath, f))
            key = (spk, stem)
            if ext == ".wav":
                audio_paths[key] = full
            elif ext == ".mpg":
                video_paths[key] = full

    #throw away anything that doesn't have both audio and video
    common = set(audio_paths) & set(video_paths)
    by_speaker = {}
    for key in common:
        spk = key[0]
        if spk not in by_speaker:
            by_speaker[spk] = []
        by_speaker[spk].append({"audio": audio_paths[key], "video": video_paths[key]})
    for spk in by_speaker:
        by_speaker[spk].sort(key=lambda x: x["audio"])
    return by_speaker

def split_by_speaker(by_speaker, train_spk, valid_spk, test_spk):
    train_list = []
    valid_list = []
    test_list = []
    for spk, pairs in by_speaker.items():
        if spk in train_spk:
            train_list.extend(pairs)
        elif spk in valid_spk:
            valid_list.extend(pairs)
        elif spk in test_spk:
            test_list.extend(pairs)
    return train_list, valid_list, test_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_root", required=True)
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()

    train_spk = [f"s{i}" for i in range(1, 29)]
    valid_spk = [f"s{i}" for i in range(29, 32)]
    test_spk = [f"s{i}" for i in range(32, 35)]

    by_speaker = collect_pairs(args.grid_root)
    if not by_speaker:
        return

    train_list, valid_list, test_list = split_by_speaker(by_speaker, train_spk, valid_spk, test_spk)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    save_json(train_list, os.path.join(out_dir, "grid_train.json"))
    save_json(valid_list, os.path.join(out_dir, "grid_valid.json"))
    save_json(test_list, os.path.join(out_dir, "grid_test.json"))

if __name__ == "__main__":
    main()
