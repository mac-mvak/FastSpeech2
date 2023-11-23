from hw_sg.utils.parse_config import ConfigParser
from hw_sg.utils.object_loading import get_dataloaders
import hw_sg.model as module_arch
import waveglow
import os
import torch
import hw_sg.text as text
from tqdm import tqdm
import numpy as np
from waveglow.inference import get_wav

import torchaudio
import utils

import json


WaveGlow = utils.get_WaveGlow()
WaveGlow = WaveGlow.cuda()


with open('/home/mac-mvak/code_disk/FastSpeech2/saved/models/default_config/1123_115608/config.json') as f:
    cfg = json.load(f)


cfg = ConfigParser(cfg)

device = torch.device('cuda:0')
model = cfg.init_obj(cfg["arch"], module_arch)
model.load_state_dict(torch.load(
    '/home/mac-mvak/code_disk/FastSpeech2/saved/models/default_config/1123_115608/checkpoint-epoch6.pth', map_location='cuda:0')["state_dict"])
model = model.to(device)
model = model.eval()

def synthesis(model, text, length_coeff=1.0, 
              energy_coeff=1.0, pitch_coeff=1.0):
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
text_cleaners = ['english_cleaners']

def get_data():
    tests = [ 
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space"
    ]
    data_list = list(text.text_to_sequence(test, text_cleaners) for test in tests)

    return data_list

data_list = get_data()
for u in [0.8, 0.9, 1., 1.1, 1.2]:
    for p in [(u, 1., 1.), (1., u, 1.), (1.,1.,u)]:
        speed, energy, pitch = p
        for i, phn in tqdm(enumerate(data_list)):
            mel, mel_cuda = synthesis(model, phn, length_coeff=speed,
                                          energy_coeff=energy,
                                          pitch_coeff=pitch)
        
            os.makedirs("results", exist_ok=True)
        
            audio = get_wav(mel_cuda, WaveGlow, sampling_rate=22050)
            torchaudio.save(f"results/s={speed}_e={energy}_p={pitch}_{i}_waveglow.wav", audio.unsqueeze(0), 22050)

