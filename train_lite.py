
import argparse
import os
import time
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader
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
    safe_backward,
    check_loss_health,
)

torch.backends.cudnn.benchmark = True

def setup_optimizer(generator, cfg):
    #3group differential lr: audio/visual_enc/fusion
    lr=cfg['training_cfg']['learning_rate']
    betas=(cfg['training_cfg']['adam_b1'],cfg['training_cfg']['adam_b2'])
    use_diff_lr=cfg['training_cfg'].get('use_differential_lr',False)
    if use_diff_lr:
        audioModules=['dense_encoder','TSMamba','mask_decoder','phase_decoder']
        visEncModules=['visualEncoder']
        audio_lr=cfg['training_cfg'].get('audio_lr_scale',1.0)
        vis_lr=cfg['training_cfg'].get('visual_lr_scale',0.2)
        audio_params,vis_enc_params,fusion_params=[],[],[]
        for name,param in generator.named_parameters():
            if param.requires_grad:
                if any(name.startswith(m) or name.startswith(f'module.{m}') for m in audioModules):
                    audio_params.append(param)
                elif any(name.startswith(m) or name.startswith(f'module.{m}') for m in visEncModules):
                    vis_enc_params.append(param)
                else:
                    fusion_params.append(param)
        param_groups=[
            {'params':audio_params,'lr':lr*audio_lr},
            {'params':vis_enc_params,'lr':lr*vis_lr},
            {'params':fusion_params,'lr':lr},
        ]
        return optim.AdamW(param_groups,betas=betas)
    else:
        trainable_params = [p for p in generator.parameters() if p.requires_grad]
        return optim.AdamW(trainable_params, lr=lr, betas=betas)

def setup_scheduler(optim_g, cfg):
    lr_decay = cfg['training_cfg']['lr_decay']
    return ExponentialLR(optim_g, gamma=lr_decay)

def create_dataset(cfg,train=True,split=True):
    datasetType=cfg['data_cfg'].get('dataset_type','grid')  
    if train:
        dataJson=cfg['data_cfg']['train_data_json']
        noiseJson=cfg['data_cfg'].get('train_noise_json',None)
    else:
        dataJson=cfg['data_cfg']['valid_data_json']
        noiseJson=cfg['data_cfg'].get('valid_noise_json',None)
    visCfg=cfg.get('visual_cfg',{})
    snrRange=cfg['training_cfg'].get('snr_range',[-5,20])
    rirJson=cfg['data_cfg'].get('rir_json',None) if train else None
    rirProb=cfg['data_cfg'].get('rir_prob',0.3)
    common=dict(
        data_json=dataJson,noise_json=noiseJson,
        sampling_rate=cfg['stft_cfg']['sampling_rate'],
        segment_size=cfg['training_cfg']['segment_size'],
        n_fft=cfg['stft_cfg']['n_fft'],
        hop_size=cfg['stft_cfg']['hop_size'],
        win_size=cfg['stft_cfg']['win_size'],
        compress_factor=cfg['model_cfg']['compress_factor'],
        snr_range=tuple(snrRange),
        face_size=visCfg.get('face_size',96),
        video_fps=visCfg.get('video_fps',25),
        split=split,shuffle=train,
        rir_json=rirJson,rir_prob=rirProb)

    if datasetType=='grid':
        from dataloaders.dataloader_grid import GRIDAVDataset
        visAug=train and cfg.get('training_cfg',{}).get('visual_augmentation',False)
        return GRIDAVDataset(visual_augmentation=visAug,**common)
    elif datasetType=='lrs2':
        from dataloaders.dataloader_lrs import LRSAVDataset
        visDegProb=cfg.get('training_cfg',{}).get('visual_degradation_prob',0.0) if train else 0.0
        modConflict=cfg.get('training_cfg',{}).get('modality_conflict_prob',0.0) if train else 0.0
        cocktailProb=cfg.get('training_cfg',{}).get('cocktail_party_prob',0.0) if train else 0.0
        return LRSAVDataset(visual_degradation_prob=visDegProb,
                            modality_conflict_prob=modConflict,
                            cocktail_party_prob=cocktailProb,**common)
    elif datasetType=='vox':
        from dataloaders.dataloader_vox import VoxCelebAVDataset
        minAudioLen=cfg['data_cfg'].get('min_audio_len',8000)
        return VoxCelebAVDataset(min_audio_len=minAudioLen,**common)
    else:
        raise ValueError(f"Unknown dataset_type '{datasetType}'")


def create_dataloader(dataset, cfg, train=True, rank=0):
    batch_size = cfg['training_cfg']['batch_size'] if train else 1
    num_workers = cfg['env_setting']['num_workers'] if train else 1
    num_gpus = cfg['env_setting']['num_gpus']

    sampler = None
    shuffle = train
    
    if num_gpus>1 and train:
        if batch_size < num_gpus:
            raise ValueError(
                f"Batch size must be greater than or equal to the number of GPUs")
        batch_size = batch_size // num_gpus
        sampler = DistributedSampler(
            dataset,
            num_replicas=num_gpus,
            rank=rank,
            shuffle=train,
        )
        shuffle = False

    return DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=train,
        sampler = sampler
    )


def load_latest_generator_state(exp_path, device, generator, cfg, optim_g=None, scheduler_g=None):
    num_gpus=cfg['env_setting']['num_gpus']
    ckpt_path = scan_checkpoint(exp_path, "g_")
    steps, start_epoch, best_pesq = 0, 0, -1.0
    if ckpt_path is None:
        return steps, start_epoch, best_pesq, ckpt_path

    state = load_checkpoint(ckpt_path, device)

    generator.load_state_dict(state["generator"], strict=False)

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


def train(rank, args, cfg):
    num_gpus=cfg['env_setting']['num_gpus']
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']

    if num_gpus>1:
        initialize_process_group(cfg,rank)
    device=torch.device(f'cuda:{rank}')

    generator = LiteAVSEMamba(cfg).to(device)
    optim_g = setup_optimizer(generator, cfg)
    
    scheduler_g = setup_scheduler(optim_g, cfg)

    steps, start_epoch, best_pesq, _ = load_latest_generator_state(
        args.exp_path, device, generator, cfg, optim_g, scheduler_g
    )
    if num_gpus>1:
        generator=DistributedDataParallel(generator,device_ids=[rank]).to(device)

    model_ref = generator.module if num_gpus>1 else generator
    trainable_params = [p for p in model_ref.parameters() if p.requires_grad]

    train_loader = create_dataloader(create_dataset(cfg, train=True), cfg, train=True, rank=rank)
    if rank==0:
        validation_loader = create_dataloader(create_dataset(cfg, train=False), cfg, train=False)
        sw=SummaryWriter(os.path.join(args.exp_path,'logs'))

    nan_patience = int(cfg['training_cfg'].get('nan_patience', 3))
    consecutive_bad_batches = 0
    generator.train()

    for epoch in range(max(0,start_epoch), cfg['training_cfg']['training_epochs']):
        if rank==0:
            epoch_start = time.time()
        print(f"Epoch {epoch + 1}")
        if num_gpus>1:
            train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            if rank==0:
                step_start = time.time()
            
            if len(batch)==8:
                clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha, video, visual_deg = batch
                visual_deg = visual_deg.to(device, non_blocking=True)
            else:
                clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha, video = batch
                visual_deg = None

            clean_audio = clean_audio.to(device, non_blocking=True)
            clean_mag = clean_mag.to(device, non_blocking=True)
            clean_pha = clean_pha.to(device, non_blocking=True)
            clean_com = clean_com.to(device, non_blocking=True)
            noisy_mag = noisy_mag.to(device, non_blocking=True)
            noisy_pha = noisy_pha.to(device, non_blocking=  True)
            video = video.to(device, non_blocking=True)

            model_ref.currentStep = steps
            mag_g, pha_g, com_g, intermediates = generator(noisy_mag, noisy_pha, video, return_intermediates=True, visual_degraded=visual_deg)
            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)

            optim_g.zero_grad(set_to_none=True)
            loss_mag = F.mse_loss(clean_mag, mag_g)
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, cfg)
            loss_pha = loss_ip + loss_gd + loss_iaf
            loss_com = F.mse_loss(clean_com, com_g) * 2.0
            _, _, rec_com = mag_phase_stft(audio_g, n_fft, hop_size, win_size, compress_factor, addeps=True)
            loss_con = F.mse_loss(com_g, rec_com) * 2.0
            loss_sisdr = -si_sdr_loss(clean_audio, audio_g)
            loss_time = F.l1_loss(clean_audio, audio_g)

            loss_gen_all = (
                loss_mag * cfg['training_cfg']['loss']['magnitude'] +
                loss_pha * cfg['training_cfg']['loss']['phase'] +
                loss_com * cfg['training_cfg']['loss']['complex'] +
                loss_con * cfg['training_cfg']['loss']['consistancy'] +
                loss_sisdr * cfg['training_cfg']['loss']['si_sdr'] +
                loss_time * cfg['training_cfg']['loss']['time']
            )

            aux_visual=cfg.get('training_cfg',{}).get('loss',{}).get('aux_visual',0.0)
            if aux_visual>0:
                loss_aux=intermediates.get('aux_loss',torch.tensor(0.0,device=device))
                loss_gen_all=loss_gen_all+loss_aux*aux_visual

            alpha_smooth=cfg.get('training_cfg',{}).get('loss',{}).get('alpha_smooth',0.0)
            if alpha_smooth>0 and 'alpha' in intermediates:
                alpha=intermediates['alpha']
                alphaDiff=(alpha[:,1:,:]-alpha[:,:-1,:]).abs()
                loss_alpha_smooth=alphaDiff.mean()
                sat_thresh=cfg.get('training_cfg',{}).get('loss',{}).get('alpha_sat_threshold',0.4)
                alpha_sat=((alpha-0.5).abs()-sat_thresh).clamp(min=0).mean()
                loss_gen_all=loss_gen_all+(loss_alpha_smooth+alpha_sat*0.5)*alpha_smooth

            alpha_entropy=cfg.get('training_cfg',{}).get('loss',{}).get('alpha_entropy',0.0)
            if alpha_entropy>0 and 'alpha' in intermediates:
                a=intermediates['alpha'].clamp(1e-6,1-1e-6)
                entropy=-(a*a.log()+(1-a)*(1-a).log()).mean()
                loss_gen_all=loss_gen_all-entropy*alpha_entropy

            gate_supervision =cfg.get('training_cfg',{}).get('loss',{}).get('gate_supervision',0.0)
            if gate_supervision>0 and 'gate_loss' in intermediates:
                loss_gen_all=loss_gen_all+intermediates['gate_loss']*gate_supervision

            is_valid, consecutive_bad_batches, should_reload = check_loss_health(loss_gen_all, consecutive_bad_batches, nan_patience)
            if not is_valid:
                print(f"Steps {steps}: invalid loss detected, skipping batch.")
                if should_reload:
                    print("Reloading latest checkpoint after repeated invalid losses.")
                    steps, start_epoch, best_pesq, _ = load_latest_generator_state(
                        args.exp_path, device, model_ref, cfg, optim_g, scheduler_g
                    )
                    consecutive_bad_batches = 0
                continue

            if not safe_backward(loss_gen_all, optim_g):
                consecutive_bad_batches += 1
                continue

            clip_grad_norm_(trainable_params, max_norm=1.0)
            optim_g.step()
            consecutive_bad_batches = 0

            if rank==0:
                if steps % cfg['env_setting']['stdout_interval'] == 0:
                    gate_info = ""
                    if 'alpha' in intermediates:
                        alpha = intermediates['alpha']
                        gate_info += ", alpha: {:4.3f}+/-{:4.3f}".format(
                            alpha.mean().item(),
                            alpha.std().item(),
                        )
                    if 'freq_gate' in intermediates:
                        freq_gate = intermediates['freq_gate']
                        gate_info += ", fg: {:4.3f}+/-{:4.3f}".format(
                            freq_gate.mean().item(),
                            freq_gate.std().item(),
                        )
                    if 'gate_loss' in intermediates:
                        gate_info += ", gate_loss: {:4.3f}".format(
                            intermediates['gate_loss'].item()
                        )
                    print(
                        "Steps: {:d}, Loss: {:4.3f}, Mag: {:4.3f}, Pha: {:4.3f}, Com: {:4.3f}, "
                        "Con: {:4.3f}, SI-SDR: {:4.3f}, Time: {:4.3f}, s/b: {:4.3f}{}".format(
                            steps,
                            loss_gen_all.item(),
                            loss_mag.item(),
                            loss_pha.item(),
                            loss_com.item(),
                            loss_con.item(),
                            loss_sisdr.item(),
                            loss_time.item(),
                            time.time() - step_start,
                            gate_info,
                        )
                    )
                
                if steps % cfg['env_setting']['summary_interval'] == 0 and steps != 0:
                    sw.add_scalar("Training/Generator Loss",loss_gen_all.item(),steps)
                    sw.add_scalar("Training/Magnitude Loss",loss_mag.item(),steps)
                    sw.add_scalar("Training/Phase Loss",loss_pha.item(),steps)
                    sw.add_scalar("Training/Time Loss",loss_time.item(),steps)
                    if 'alpha' in intermediates:
                        sw.add_scalar("Gates/alpha_mean",intermediates['alpha'].mean().item(),steps)
                        sw.add_scalar("Gates/alpha_std",intermediates['alpha'].std().item(),steps)
                    if 'freq_gate' in intermediates:
                        sw.add_scalar("Gates/fsvg_mean",intermediates['freq_gate'].mean().item(),steps)
                    if 'gate_loss' in intermediates:
                        sw.add_scalar("Gates/gate_supervision",intermediates['gate_loss'].item(),steps)

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
                    if metrics['pesq'] >= best_pesq:
                        best_pesq = metrics['pesq']
                        
                        checkpoint = {
                            "generator": model_ref.state_dict(),
                            "optim_g": optim_g.state_dict(),
                            "scheduler_g": scheduler_g.state_dict(),
                            "steps": steps,
                            "epoch": epoch,
                            "best_pesq": best_pesq,
                        }
                        save_checkpoint(
                            os.path.join(args.exp_path, f"g_{steps:08d}.pth"),
                            checkpoint,
                        )
                        save_checkpoint(
                            os.path.join(args.exp_path, "best_g.pth"),
                            checkpoint,
                        )

                if steps % cfg['env_setting']['checkpoint_interval'] == 0 and steps != 0:
                    save_checkpoint(
                        os.path.join(args.exp_path, f"g_{steps:08d}.pth"),
                        {
                            "generator": model_ref.state_dict(),
                            "optim_g": optim_g.state_dict(),
                            "scheduler_g": scheduler_g.state_dict(),
                            "steps": steps,
                            "epoch": epoch,
                            "best_pesq": best_pesq,
                        },
                    )

            steps += 1

        scheduler_g.step()
        if rank==0:
            print(f"Epoch {epoch + 1} finished in {int(time.time() - epoch_start)} sec.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='exp')
    parser.add_argument('--exp_name', default=None)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg['env_setting']['seed']
    num_gpus = cfg['env_setting']['num_gpus']
    available_gpus = torch.cuda.device_count()

    if args.exp_name is None:
        args.exp_name = os.path.splitext(os.path.basename(args.config))[0]

    initialize_seed(seed)
    args.exp_path = os.path.join(args.exp_folder, args.exp_name)
    build_env(args.config, 'config.yaml', args.exp_path)

    if torch.cuda.is_available():
        if num_gpus > available_gpus:
            raise ValueError(f"Not enough GPUs available")
        print_gpu_info(num_gpus, cfg)
    else:
        raise RuntimeError("LiteAVSE training requires CUDA.")
    if num_gpus>1:
        mp.spawn(train,nprocs=num_gpus,args=(args,cfg))
    else:
        train(0,args,cfg)


if __name__ == "__main__":
    main()
