#author:Billy
#LiteAVSEMamba-RealTime inference
import os
import argparse
import torch
import librosa
import soundfile as sf
from models.stfts import mag_phase_stft,mag_phase_istft
from models.generator import LiteAVSEMamba
from dataloaders.av_utils import load_video_frames
from utils.util import load_config

def inference(args,device):
    cfg=load_config(args.config)
    nFft=cfg['stft_cfg']['n_fft']
    hopSize=cfg['stft_cfg']['hop_size']
    winSize=cfg['stft_cfg']['win_size']
    compress=cfg['model_cfg']['compress_factor']
    sr=cfg['stft_cfg']['sampling_rate']
    visCfg=cfg.get('visual_cfg',{})
    faceSize=visCfg.get('face_size',96)
    fps=visCfg.get('video_fps',25)
    model=LiteAVSEMamba(cfg).to(device)
    state=torch.load(args.checkpoint_file,map_location=device)
    if 'generator' in state:
        state=state['generator']
    model.load_state_dict(state)
    model.eval()
    os.makedirs(args.output_folder,exist_ok=True)
    with torch.no_grad():
        if args.input_video is not None:
            #single file mode
            noisyWav,_=librosa.load(args.input_audio,sr=sr)
            noisyWav=torch.FloatTensor(noisyWav).to(device)
            video=load_video_frames(args.input_video,face_size=faceSize,fps=fps)
            video=video.unsqueeze(0).to(device)
            norm=torch.sqrt(len(noisyWav)/torch.sum(noisyWav**2.0)).to(device)
            noisyWav=(noisyWav*norm).unsqueeze(0)
            nMag,nPha,nCom=mag_phase_stft(noisyWav,nFft,hopSize,winSize,compress)
            eMag,ePha,eCom=model(nMag,nPha,video)
            out=mag_phase_istft(eMag,ePha,nFft,hopSize,winSize,compress)
            out=out/norm
            fname=os.path.basename(args.input_audio)
            outPath=os.path.join(args.output_folder,fname)
            sf.write(outPath,out.squeeze().cpu().numpy(),sr,'PCM_16')
            print(f"Saved: {outPath}")

        elif args.input_folder is not None:
            #folder mode, paired .wav+.mp4
            for fname in os.listdir(args.input_folder):
                if not fname.endswith('.wav'):
                    continue
                baseName=os.path.splitext(fname)[0]
                audioPath=os.path.join(args.input_folder,fname)
                videoPath=os.path.join(args.input_folder,baseName+'.mp4')
                print(f"Processing: {fname}")
                noisyWav,_=librosa.load(audioPath,sr=sr)
                noisyWav=torch.FloatTensor(noisyWav).to(device)
                norm=torch.sqrt(len(noisyWav)/torch.sum(noisyWav**2.0)).to(device)
                noisyWav=(noisyWav*norm).unsqueeze(0)
                nMag,nPha,nCom=mag_phase_stft(noisyWav,nFft,hopSize,winSize,compress)
                video=None
                if os.path.exists(videoPath):
                    video=load_video_frames(videoPath,face_size=faceSize,fps=fps)
                    video=video.unsqueeze(0).to(device)
                eMag,ePha,eCom=model(nMag,nPha,video)
                out=mag_phase_istft(eMag,ePha,nFft,hopSize,winSize,compress)
                out=out/norm

                outPath=os.path.join(args.output_folder,fname)
                sf.write(outPath,out.squeeze().cpu().numpy(),sr,'PCM_16')
                print(f"Saved: {outPath}")

def main():
    print('LiteAVSEMamba inference')
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_folder',default=None)
    parser.add_argument('--input_audio',default=None)
    parser.add_argument('--input_video',default=None)
    parser.add_argument('--output_folder',default='results_av')
    parser.add_argument('--config',required=True)
    parser.add_argument('--checkpoint_file',required=True)
    args=parser.parse_args()
    if args.input_folder is None and (args.input_audio is None or args.input_video is None):
        raise ValueError("Provide --input_folder or both --input_audio and --input_video")

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference(args,device)

if __name__=='__main__':
    main()
