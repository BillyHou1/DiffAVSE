# Author: Fan
# Scan noise directories, merge into one pool, split train/valid (~5% held out).

import os
import random
import argparse
from utils import save_json


def scan_noise_dirs(noise_dirs, extensions=('.wav', '.flac')):
    """
    Recursively scan directories for audio files.

    Args:
        noise_dirs:  list of str, directories to scan
        extensions:  tuple of str, file extensions to include
    Returns:
        list of absolute file paths
    """
    out = []
    for noise_dir in noise_dirs:
        noise_dir = os.path.abspath(noise_dir)
        if not os.path.isdir(noise_dir):
            continue
        for root, dirs, files in os.walk(noise_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in extensions:
                    full = os.path.abspath(os.path.join(root, file))
                    out.append(full)
    return out


def split_noise(file_list, val_ratio=0.05, seed=1234):
    """
    Split noise files into train and validation sets.

    Args:
        file_list: list of str, file paths
        val_ratio: float, fraction to hold out
        seed:      int, for reproducibility
    Returns:
        (train_list, valid_list)
    """
    random.seed(seed)
    random.shuffle(file_list)
    n = len(file_list)
    n_val = max(1, int(n * val_ratio))
    valid_list = file_list[:n_val]
    train_list = file_list[n_val:]
    return train_list, valid_list



def main():
    parser = argparse.ArgumentParser(description='Prepare noise pool from multiple sources')
    parser.add_argument('--noise_dirs', nargs='+', required=True,
                        help='directories containing noise audio files')
    parser.add_argument('--output_dir', default='data', help='output directory for JSON files')
    args = parser.parse_args()

    noise_dirs = args.noise_dirs
    file_list = scan_noise_dirs(noise_dirs)
    train_list, valid_list = split_noise(file_list)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    save_json(train_list, os.path.join(out_dir, "noise_train.json"))
    save_json(valid_list, os.path.join(out_dir, "noise_valid.json"))
    print("train:", len(train_list), "valid:", len(valid_list))

if __name__ == '__main__':
    main()
