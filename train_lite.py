# Ronny (main loop) + Fan (data loading) + Billy (support)
# Training script for LiteAVSEMamba, forward pass, loss, backprop, validate,
# checkpoint.
#
# TODO implement the training loop test
#
# 6 losses, 5 active + 1 disabled: mag 0.9 | phase 0.3 | complex 0.1 |
# consistency 0.1 | SI-SDR 0.3 | time 0.0. You still have to compute
# loss_time = F.l1_loss(clean_audio, audio_g) even though its weight is 0.0,
# the config key lookup crashes if you skip it. Complex and consistency are
# both scaled x2 internally, loss_com = F.mse_loss(...) * 2 etc.
#
# Generator-only training with AdamW, no discriminator. Only include params
# with requires_grad=True cause visual encoder backbone is frozen. Scheduler
# is ExponentialLR with gamma from config.
#
# Validate every N steps, compute PESQ/STOI/SI-SDR on full val set with
# torch.no_grad. Save best model when PESQ beats previous best.
# Checkpoints: g_{step:08d}.pth
#
# NaN safety: check_loss_health to skip bad batches and reload from last
# checkpoint after too many consecutive NaNs, safe_backward with try/except
# around loss.backward(), gradient clipping max_norm=1.0. See utils/util.py.
#
# Dataloader returns 7-tuple clean_audio, clean_mag, clean_pha, clean_com,
# noisy_mag, noisy_pha, video. Move all to GPU with non_blocking=True.
        
import argparse
import os
import time
import warnings
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DistributedSampler, DataLoader
from dataloaders.dataloader_av import AVDataset
from models.generator import LiteAVSEMamba
from models.stfts import mag_phase_istft, mag_phase_stft
from models.loss import phase_losses, pesq_score, si_sdr_loss, si_sdr_score, stoi_score
from utils.util import (
    build_env,
    initialize_seed,
    load_checkpoint,
    load_config,
    print_gpu_info,
    save_checkpoint,
    scan_checkpoint,
    initialize_process_group,
)

torch.backends.cudnn.benchmark = True


def check_loss_health(loss):
    return torch.isfinite(loss).all().item() and (not torch.isnan(loss).any().item())


def safe_backward(loss):
    try:
        loss.backward()
        return True
    except RuntimeError as exc:
        print(f"Backward failed: {exc}")
        return False


def create_dataset(cfg, train=True):
    data_json = cfg['data_cfg']['train_data_json'] if train else cfg['data_cfg']['valid_data_json']
    noise_json = cfg['data_cfg']['train_noise_json'] if train else cfg['data_cfg']['valid_noise_json']
    return AVDataset(
        data_json=data_json,
        noise_json=noise_json,
        cfg=cfg,
        split=train,
        visual_augmentation=train,
    )


def create_dataloader(dataset, cfg, train=True):
    if cfg['env_setting']['num_gpus'] > 1:
        sampler = DistributedSampler(dataset)
        sampler.set_epoch(cfg['training_cfg']['training_epochs'])
        batch_size = (cfg['training_cfg']['batch_size'] // cfg['env_setting']['num_gpus']) if train else 1
    else:
        sampler = None
        batch_size = cfg['training_cfg']['batch_size'] if train else 1
    num_workers = cfg['env_setting']['num_workers'] if train else 1
    return DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=(sampler is None) and train,
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True if train else False
    )


def load_latest_generator_state(exp_path, device, generator, optim_g=None, scheduler_g=None):
    ckpt_path = scan_checkpoint(exp_path, "g_")
    steps, start_epoch, best_pesq = 0, 0, -1.0
    if ckpt_path is None:
        return steps, start_epoch, best_pesq, ckpt_path

    state = load_checkpoint(ckpt_path, device)
    generator.load_state_dict(state['generator'], strict=False)
    if optim_g is not None and 'optim_g' in state:
        optim_g.load_state_dict(state['optim_g'])
    if scheduler_g is not None and 'scheduler_g' in state:
        scheduler_g.load_state_dict(state['scheduler_g'])
    steps = int(state.get('steps', 0))
    start_epoch = int(state.get('epoch', 0))
    best_pesq = float(state.get('best_pesq', -1.0))
    return steps, start_epoch, best_pesq, ckpt_path


def validate(generator, validation_loader, cfg, device):
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']

    generator.eval()
    audios_r, audios_g = [], []
    val_mag_err_tot = 0.0
    val_pha_err_tot = 0.0
    val_com_err_tot = 0.0

    with torch.no_grad():
        count = 0
        for count, batch in enumerate(validation_loader, start=1):
            clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha, video = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            clean_mag = clean_mag.to(device, non_blocking=True)
            clean_pha = clean_pha.to(device, non_blocking=True)
            clean_com = clean_com.to(device, non_blocking=True)
            noisy_mag = noisy_mag.to(device, non_blocking=True)
            noisy_pha = noisy_pha.to(device, non_blocking=True)
            video = video.to(device, non_blocking=True)

            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha, video)
            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)

            audios_r += torch.split(clean_audio, 1, dim=0)
            audios_g += torch.split(audio_g, 1, dim=0)

            val_mag_err_tot += F.mse_loss(clean_mag, mag_g).item()
            val_ip, val_gd, val_iaf = phase_losses(clean_pha, pha_g, cfg)
            val_pha_err_tot += (val_ip + val_gd + val_iaf).item()
            val_com_err_tot += F.mse_loss(clean_com, com_g).item()

    generator.train()
    if count == 0:
        return {
            "pesq": -1.0,
            "stoi": -1.0,
            "si_sdr": -99.0,
            "mag": 0.0,
            "pha": 0.0,
            "com": 0.0,
        }

    return {
        "pesq": float(pesq_score(audios_r, audios_g, cfg)),
        "stoi": float(stoi_score(audios_r, audios_g, cfg)),
        "si_sdr": float(si_sdr_score(audios_r, audios_g)),
        "mag": val_mag_err_tot / count,
        "pha": val_pha_err_tot / count,
        "com": val_com_err_tot / count,
    }


def train(args, cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("LiteAVSE training requires CUDA.")

    device = torch.device('cuda:0')
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']

    generator = LiteAVSEMamba(cfg).to(device)
    trainable_params = [p for p in generator.parameters() if p.requires_grad]
    optim_g = AdamW(
        trainable_params,
        lr=cfg['training_cfg']['learning_rate'],
        betas=(cfg['training_cfg']['adam_b1'], cfg['training_cfg']['adam_b2']),
    )
    scheduler_g = ExponentialLR(optim_g, gamma=cfg['training_cfg']['lr_decay'])

    steps, start_epoch, best_pesq, _ = load_latest_generator_state(
        args.exp_path, device, generator, optim_g, scheduler_g
    )

    train_loader = create_dataloader(create_dataset(cfg, train=True), cfg, train=True)
    validation_loader = create_dataloader(create_dataset(cfg, train=False), cfg, train=False)

    nan_patience = int(cfg['training_cfg'].get('nan_patience', 3))
    consecutive_bad_batches = 0
    generator.train()

    for epoch in range(max(0,start_epoch), cfg['training_cfg']['training_epochs']):
        epoch_start = time.time()
        print(f"Epoch {epoch + 1}")
        for batch in train_loader:
            step_start = time.time()
            clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha, video = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            clean_mag = clean_mag.to(device, non_blocking=True)
            clean_pha = clean_pha.to(device, non_blocking=True)
            clean_com = clean_com.to(device, non_blocking=True)
            noisy_mag = noisy_mag.to(device, non_blocking=True)
            noisy_pha = noisy_pha.to(device, non_blocking=True)
            video = video.to(device, non_blocking=True)

            optim_g.zero_grad(set_to_none=True)
            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha, video)
            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)

            loss_mag = F.mse_loss(clean_mag, mag_g)
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, cfg)
            loss_pha = loss_ip + loss_gd + loss_iaf
            loss_com = F.mse_loss(clean_com, com_g) * 2.0
            _, _, rec_com = mag_phase_stft(
                audio_g, n_fft, hop_size, win_size, compress_factor, addeps=True
            )
            loss_con = F.mse_loss(com_g, rec_com) * 2.0
            loss_sisdr = -si_sdr_loss(clean_audio, audio_g)
            loss_time = F.l1_loss(clean_audio, audio_g)

            loss_gen_all = (
                loss_mag * cfg['training_cfg']['loss']['magnitude'] +
                loss_pha * cfg["training_cfg"]["loss"]["phase"] +
                loss_com * cfg["training_cfg"]["loss"]["complex"] +
                loss_con * cfg["training_cfg"]["loss"]["consistancy"] +
                loss_sisdr * cfg["training_cfg"]["loss"]["si_sdr"] +
                loss_time * cfg["training_cfg"]["loss"]["time"]
            )

            if not check_loss_health(loss_gen_all):
                consecutive_bad_batches += 1
                print(f"Steps {steps}: invalid loss detected, skipping batch.")
                if consecutive_bad_batches >= nan_patience:
                    print("Reloading latest checkpoint after repeated invalid losses.")
                    _, _, best_pesq, _ = load_latest_generator_state(
                        args.exp_path, device, generator, optim_g, scheduler_g
                    )
                    consecutive_bad_batches = 0
                continue

            if not safe_backward(loss_gen_all):
                consecutive_bad_batches += 1
                continue

            clip_grad_norm_(trainable_params, max_norm=1.0)
            optim_g.step()
            consecutive_bad_batches = 0

            if steps % cfg['env_setting']['stdout_interval'] == 0:
                print(
                    "Steps: {:d}, Loss: {:4.3f}, Mag: {:4.3f}, Pha: {:4.3f}, Com: {:4.3f}, "
                    "Con: {:4.3f}, SI-SDR: {:4.3f}, Time: {:4.3f}, s/b: {:4.3f}".format(
                        steps,
                        loss_gen_all.item(),
                        loss_mag.item(),
                        loss_pha.item(),
                        loss_com.item(),
                        loss_con.item(),
                        loss_sisdr.item(),
                        loss_time.item(),
                        time.time() - step_start,
                    )
                )

            if steps % cfg['env_setting']['validation_interval'] == 0 and steps != 0:
                metrics = validate(generator, validation_loader, cfg, device)
                print(
                    "Valid @ {:d}: PESQ {:4.3f}, STOI {:4.3f}, SI-SDR {:4.3f}, Mag {:4.3f}, "
                    "Pha {:4.3f}, Com {:4.3f}".format(
                        steps,
                        metrics["pesq"],
                        metrics["stoi"],
                        metrics["si_sdr"],
                        metrics["mag"],
                        metrics["pha"],
                        metrics["com"],
                    )
                )
                if metrics["pesq"] >= best_pesq:
                    best_pesq = metrics["pesq"]
                    save_checkpoint(
                        os.path.join(args.exp_path, f"g_{steps:08d}.pth"),
                        {
                            "generator": generator.state_dict(),
                            "optim_g": optim_g.state_dict(),
                            "scheduler_g": scheduler_g.state_dict(),
                            "steps": steps,
                            "epoch": epoch,
                            "best_pesq": best_pesq,
                        },
                    )

            if steps % cfg['env_setting']['checkpoint_interval'] == 0 and steps != 0:
                save_checkpoint(
                    os.path.join(args.exp_path, f"g_{steps:08d}.pth"),
                    {
                        "generator": generator.state_dict(),
                        "optim_g": optim_g.state_dict(),
                        "scheduler_g": scheduler_g.state_dict(),
                        "steps": steps,
                        "epoch": epoch,
                        "best_pesq": best_pesq,
                    },
                )

            steps += 1

        scheduler_g.step()
        print(f"Epoch {epoch + 1} finished in {int(time.time() - epoch_start)} sec.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='exp')
    parser.add_argument('--exp_name', default='LiteAVSE')
    parser.add_argument('--config', default='recipes/LiteAVSE/LiteAVSE.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg['env_setting']['seed']
    num_gpus = cfg['env_setting']['num_gpus']
    available_gpus = torch.cuda.device_count()

    initialize_seed(seed)
    args.exp_path = os.path.join(args.exp_folder, args.exp_name)
    build_env(args.config, 'config.yaml', args.exp_path)

    if torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_available_gpus}")
        print_gpu_info(num_available_gpus, cfg)
    else:
        warnings.warn("CUDA is not available.", UserWarning)

    if num_gpus > 1:
        mp.spawn(train, nprocs=num_gpus, args=(args, cfg))
    else:
        train(args, cfg)


if __name__ == "__main__":
    main()
