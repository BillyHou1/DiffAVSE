#Author: Fan
#VoxCeleb2 json lists, split dev into train+valid
import os
import random
import argparse
from dataloaders.av_utils import save_json, extract_audio_ffmpeg as extract_audio_from_video

def collect_clips(vox2_root, subset, extract_audio=False):
    #go through dev/ or test/ and grab all the mp4 files
    root = os.path.abspath(vox2_root)
    sub_dir = os.path.join(root, subset)
    if not os.path.isdir(sub_dir):
        return []
    out = []
    for dirpath, _, filenames in os.walk(sub_dir):
        for f in filenames:
            if os.path.splitext(f)[1].lower() == ".mp4":
                full = os.path.abspath(os.path.join(dirpath, f))
                if extract_audio:
                    audio_path = extract_audio_from_video(full)
                    out.append({"audio": os.path.abspath(audio_path), "video": os.path.abspath(full)})
                else:
                    out.append({"video": os.path.abspath(full)})
    return out

def split_dev(dev_list, val_ratio=0.03, seed=1234):
    #shuffle dev and chop off a small chunk for validation
    random.seed(seed)
    lst = list(dev_list)
    random.shuffle(lst)
    n = len(lst)
    n_val = max(1, int(n * val_ratio))
    valid_list = lst[:n_val]
    train_list = lst[n_val:]
    return train_list, valid_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vox2_root", required=True)
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--val_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--extract_audio", action="store_true")
    args = parser.parse_args()
    root = os.path.abspath(args.vox2_root)
    if not os.path.isdir(root):
        return

    dev_list = collect_clips(root, "dev", args.extract_audio)
    test_list = collect_clips(root, "test", args.extract_audio)
    train_list, valid_list = split_dev(dev_list, args.val_ratio, args.seed)

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    save_json(train_list, os.path.join(out_dir, "vox_train.json"))
    save_json(valid_list, os.path.join(out_dir, "vox_valid.json"))
    save_json(test_list, os.path.join(out_dir, "vox_test.json"))

if __name__ == "__main__":
    main()
