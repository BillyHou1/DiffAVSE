# Author: Fan
# Build JSON lists for VoxCeleb2. Dev split into train/valid, test as-is.

import os
import random
import argparse
from utils import save_json, extract_audio_from_video



def collect_clips(vox2_root, subset, extract_audio=False):
    # subset = 'dev' or 'test'. walk vox2_root/dev or vox2_root/test, get all .mp4
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
    # take val_ratio of dev as valid, rest train. shuffle with seed
    random.seed(seed)
    lst = list(dev_list)
    random.shuffle(lst)
    n = len(lst)
    n_val = max(1, int(n * val_ratio))
    valid_list = lst[:n_val]
    train_list = lst[n_val:]
    return train_list, valid_list

def main():
    parser = argparse.ArgumentParser(description="VoxCeleb2 json lists")
    parser.add_argument("--vox2_root", required=True, help="VoxCeleb2 root (has dev/ and test/)")
    parser.add_argument("--output_dir", default="data", help="where to write json")
    parser.add_argument("--val_ratio", type=float, default=0.03, help="fraction of dev for valid")
    parser.add_argument("--seed", type=int, default=1234, help="random seed for split")
    parser.add_argument("--extract_audio", action="store_true", help="extract audio from video")
    args = parser.parse_args()

    root = os.path.abspath(args.vox2_root)
    if not os.path.isdir(root):
        print("Error: not a dir:", root)
        return

    dev_list = collect_clips(root, "dev", args.extract_audio)
    test_list = collect_clips(root, "test", args.extract_audio)
    train_list, valid_list = split_dev(dev_list, args.val_ratio, args.seed)

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    save_json(train_list, os.path.join(out_dir, "vox_train.json"))
    save_json(valid_list, os.path.join(out_dir, "vox_valid.json"))
    save_json(test_list, os.path.join(out_dir, "vox_test.json"))

    print("train:", len(train_list), "valid:", len(valid_list), "test:", len(test_list))
    if not dev_list and not test_list:
        print("no clips.")


if __name__ == "__main__":
    main()
