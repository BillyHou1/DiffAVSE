# Author: Fan
# Build JSON lists for LRS2 using official train/val/test splits.

import os
import argparse
from utils import save_json, extract_audio_from_video


def read_split_file(txt_path, root, extract_audio=False):
    root = os.path.abspath(root)
    if not os.path.isfile(txt_path):
        return []
    out = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            full = os.path.normpath(os.path.join(root, line))
            if extract_audio:
                audio_path = extract_audio_from_video(full)
                out.append({"audio": os.path.abspath(audio_path), "video": os.path.abspath(full)})
            else:
                out.append({"video": os.path.abspath(full)})
    return out

def main():
    parser = argparse.ArgumentParser(description="LRS2 json lists")
    parser.add_argument("--lrs2_root", required=True, help="LRS2 root dir")
    parser.add_argument("--output_dir", default="data", help="where to write json")
    parser.add_argument("--extract_audio", action="store_true", help="extract audio from video")
    args = parser.parse_args()

    root = os.path.abspath(args.lrs2_root)
    if not os.path.isdir(root):
        print("Error: not a dir:", root)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = os.path.abspath(args.output_dir)

    splits = [
        ("train", "train.txt", "lrs_train.json"),
        ("valid", "val.txt", "lrs_valid.json"),
        ("test", "test.txt", "lrs_test.json"),
    ]
    total = 0
    for name, txt_name, json_name in splits:
        txt_path = os.path.join(root, txt_name)
        lst = read_split_file(txt_path, root, args.extract_audio)
        total += len(lst)
        out_path = os.path.join(out_dir, json_name)
        save_json(lst, out_path)

    if total == 0:
        print("no entries.")


if __name__ == "__main__":
    main()
