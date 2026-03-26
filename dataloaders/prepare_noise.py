#Author: Fan
#find all noise audio files in given folders, mix them into one pool
#hold out about 5% for validation
import os
import random
import argparse
from dataloaders.av_utils import save_json

def scan_noise_dirs(noise_dirs, extensions=('.wav','.flac')):
    #walk through all the dirs and collect wav/flac paths
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
    #shuffle and split off a small validation set
    random.seed(seed)
    random.shuffle(file_list)
    n = len(file_list)
    n_val = max(1, int(n * val_ratio))
    valid_list = file_list[:n_val]
    train_list = file_list[n_val:]
    return train_list, valid_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_dirs', nargs='+', required=True)
    parser.add_argument('--output_dir', default='data')
    args = parser.parse_args()

    noise_dirs = args.noise_dirs
    file_list = scan_noise_dirs(noise_dirs)
    train_list, valid_list = split_noise(file_list)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    save_json(train_list, os.path.join(out_dir, "noise_train.json"))
    save_json(valid_list, os.path.join(out_dir, "noise_valid.json"))

if __name__ == '__main__':
    main()
