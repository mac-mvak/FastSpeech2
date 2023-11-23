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


with open('/home/mac-mvak/code_disk/FastSpeech2/saved/models/default_config/1123_090751/config.json') as f:
    cfg = json.load(f)


cfg = ConfigParser(cfg)

device = torch.device('cuda:0')
model = cfg.init_obj(cfg["arch"], module_arch)
model.load_state_dict(torch.load(
    '/home/mac-mvak/code_disk/FastSpeech2/saved/models/default_config/1123_090751/checkpoint-epoch6.pth', map_location='cuda:0')["state_dict"])
model = model.to(device)
model = model.eval()

def synthesis(model, text, alpha=1.0):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(device)
    src_pos = torch.from_numpy(src_pos).long().to(device)
    
    with torch.no_grad():
        out = model.forward(sequence, src_pos, length_coeff=alpha)
    mel = out['mel_output']
    return mel.cpu().transpose(0, 1), mel.transpose(1, 2)
text_cleaners = ['english_cleaners']

def get_data():
    tests = [ 
        "Vatoadmin has awful videos and texts. Literally the worst blogger",
        "Durian model is a very good speech synthesis!",
        "When I was twenty, I fell in love with a girl.",
        "I remove attention module in decoder and use average pooling to implement predicting r frames at once",
        "You can not improve your past, but you can improve your future. Once time is wasted, life is wasted.",
        "Death comes to all, but great achievements raise a monument which shall endure until the sun grows old."
    ]
    data_list = list(text.text_to_sequence(test, text_cleaners) for test in tests)

    return data_list

data_list = get_data()
for speed in [0.8, 1., 1.3]:
    for i, phn in tqdm(enumerate(data_list)):
        mel, mel_cuda = synthesis(model, phn, speed)
        
        os.makedirs("results", exist_ok=True)
        
        audio = get_wav(mel_cuda, WaveGlow, sampling_rate=22050)
        torchaudio.save(f"results/s={speed}_{i}_waveglow.wav", audio.unsqueeze(0), 22050)

