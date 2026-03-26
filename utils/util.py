import yaml
import torch
import os
import shutil
import glob
from torch.distributed import init_process_group

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def initialize_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def print_gpu_info(num_gpus, cfg):
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
        print('Batch size per GPU:', int(cfg['training_cfg']['batch_size'] / num_gpus))

def initialize_process_group(cfg, rank):
    init_process_group(
        backend=cfg['env_setting']['dist_cfg']['dist_backend'],
        init_method=cfg['env_setting']['dist_cfg']['dist_url'],
        world_size=cfg['env_setting']['dist_cfg']['world_size'] * cfg['env_setting']['num_gpus'],
        rank=rank
    )

def log_model_info(rank, model, exp_path):
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print("Generator Parameters :", num_params)
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'logs'), exist_ok=True)
    print("checkpoints directory :", exp_path)

def load_ckpts(args, device):
    if os.path.isdir(args.exp_path):
        cp_g = scan_checkpoint(args.exp_path, 'g_')
        cp_do = scan_checkpoint(args.exp_path, 'do_')
        if cp_g is None or cp_do is None:
            return None, None, 0, -1
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        return state_dict_g, state_dict_do, state_dict_do['steps'] + 1, state_dict_do['epoch']
    return None, None, 0, -1

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????' + '.pth')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def build_env(config, config_name, exp_path):
    os.makedirs(exp_path, exist_ok=True)
    t_path = os.path.join(exp_path, config_name)
    if config != t_path:
        shutil.copyfile(config, t_path)

def load_optimizer_states(optimizers, state_dict_do):
    if state_dict_do is not None:
        optim_g, optim_d = optimizers
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])


#crash protection and training helpers
def setup_cache_dirs(base=None):
    if base is None:
        base=os.path.join(os.path.expanduser("~"),".cache","mp")
    dirs={
        "TORCH_HOME":os.path.join(base,"torch"),
        "TRITON_CACHE_DIR":os.path.join(base,"triton"),
        "XDG_CACHE_HOME":os.path.join(base,"xdg"),
        "HF_HOME":os.path.join(base,"huggingface"),
    }
    for env_var,path in dirs.items():
        os.makedirs(path,exist_ok=True)
        os.environ[env_var]=path

def check_loss_health(loss,nan_count,threshold=5):
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        nan_count+=1
        shouldReload=nan_count>=threshold
        if shouldReload:
            print(f"[WARN] {nan_count} consecutive NaN/Inf, reloading checkpoint")
        else:
            print(f"[WARN] NaN/Inf loss ({nan_count}/{threshold}), skipping batch")
        return False,nan_count,shouldReload
    return True,0,False

def safe_backward(loss,optimizer,max_grad_norm=1.0,model_params=None):
    try:
        loss.backward()
        if model_params is not None:
            torch.nn.utils.clip_grad_norm_(model_params,max_grad_norm)
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("[WARN] OOM during backward, skipping batch")
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            return False
        raise

def safe_step(optimizer):
    try:
        optimizer.step()
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("[WARN] OOM during step, skipping batch")
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            return False
        raise

def save_best_model(exp_path,generator,num_gpus,best_pesq,step):
    filepath=os.path.join(exp_path,"best_g.pth")
    state={
        'generator':(generator.module if num_gpus>1 else generator).state_dict(),
        'best_pesq':best_pesq,
        'step':step,
    }
    print(f"Saving best model (PESQ={best_pesq:.3f} at step {step}) to {filepath}")
    torch.save(state,filepath)

def save_checkpoint_with_meta(filepath,obj,best_pesq,best_pesq_step):
    obj['best_pesq']=best_pesq
    obj['best_pesq_step']=best_pesq_step
    save_checkpoint(filepath,obj)

def load_best_pesq_from_ckpt(state_dict_do):
    if state_dict_do is None:
        return 0.0,0
    return state_dict_do.get('best_pesq',0.0),state_dict_do.get('best_pesq_step',0)
