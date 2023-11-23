import json
import logging
import os
import shutil
from curses.ascii import isascii
import numpy as np
import time
import torch
from pathlib import Path
from hw_sg.text import text_to_sequence
import torchaudio
from torch.utils.data import Dataset
from hw_sg.utils import ROOT_PATH
from tqdm import tqdm

logger = logging.getLogger(__name__)


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def get_data_to_buffer(data_path, mel_ground_truth, alignment_path, energies_path, text_cleaners, batch_expand_size):
    buffer = list()
    text = process_text(data_path)

    start = time.perf_counter()
    for i in tqdm(range(len(text))):

        mel_gt_name = os.path.join(
            mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(
            alignment_path, str(i)+".npy"))
        character = text[i][0:len(text[i])-1]
        character = np.array(
            text_to_sequence(character, text_cleaners))
        energy_name = os.path.join(
            energies_path, "ljspeech-energy-%05d.npy" % (i+1))
        energy = np.load(energy_name)

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)
        energy = torch.from_numpy(energy)

        buffer.append({"text": character, "duration": duration,
                       "mel_target": mel_gt_target,
                       "energy": energy,
                       'batch_expand_size': batch_expand_size})

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer

class LJspeechDataset(Dataset):
    def __init__(self, data_path, mel_ground_truth, alignment_path, energies_path,
                  text_cleaners, batch_expand_size, *args, **kwargs):
        buffer = get_data_to_buffer(data_path, mel_ground_truth,
                                     alignment_path, energies_path, text_cleaners, batch_expand_size)
        self.buffer = buffer
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]
