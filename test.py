import argparse

from hw_sg.utils.parse_config import ConfigParser
import hw_sg.model as module_arch
import os
import torch
import hw_sg.text as text
from tqdm import tqdm
import numpy as np
from waveglow.inference import get_wav

import torchaudio
import utils

import json


def synthesis(model, text, length_coeff=1.0, 
              energy_coeff=1.0, pitch_coeff=1.0, 
              device='cpu'):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(device)
    src_pos = torch.from_numpy(src_pos).long().to(device)
    
    with torch.no_grad():
        out = model.forward(sequence, src_pos, length_coeff=length_coeff, 
                            energy_coeff=energy_coeff,
                            pitch_coeff=pitch_coeff)
    mel = out['mel_output']
    return mel.cpu().transpose(0, 1), mel.transpose(1, 2)

def get_data():
    text_cleaners = ['english_cleaners']
    tests = [ 
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space"
    ]
    data_list = list(text.text_to_sequence(test, text_cleaners) for test in tests)

    return data_list

def main(config_pth, model_pth):
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    WaveGlow = utils.get_WaveGlow()
    WaveGlow = WaveGlow.cuda()


    with open(config_pth) as f:
        cfg = json.load(f)


    cfg = ConfigParser(cfg)

    model = cfg.init_obj(cfg["arch"], module_arch)
    model.load_state_dict(torch.load(
        model_pth, map_location=device)["state_dict"])
    model = model.to(device)
    model = model.eval()






    data_list = get_data()
    for u in [0.8, 0.9, 1., 1.1, 1.2]:
        for p in [(u, 1., 1.), (1., u, 1.), (1.,1.,u)]:
            speed, energy, pitch = p
            for i, phn in tqdm(enumerate(data_list)):
                mel, mel_cuda = synthesis(model, phn, length_coeff=speed,
                                          energy_coeff=energy,
                                          pitch_coeff=pitch,
                                          device=device)
        
                os.makedirs("results", exist_ok=True)
        
                audio = get_wav(mel_cuda, WaveGlow, sampling_rate=22050)
                torchaudio.save(f"results/s={speed}_e={energy}_p={pitch}_{i}_waveglow.wav", audio.unsqueeze(0), 22050)



if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="/home/mac-mvak/code_disk/FastSpeech2/saved/models/default_config/1123_115608/config.json",
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default="/home/mac-mvak/code_disk/FastSpeech2/saved/models/default_config/1123_115608/checkpoint-epoch6.pth",
        type=str,
        help="path to latest checkpoint (default: None)",
    )

    # custom cli options to modify configuration from default values given in json file.
    args = args.parse_args()

    main(args.config, args.resume)
