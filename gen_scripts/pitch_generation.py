from tqdm import tqdm
from pathlib import Path

import numpy as np
from scipy import interpolate
import torch
import torchaudio
import pyworld as pw



DATA_PATH = Path('data')
ENERGY_PATH = DATA_PATH / 'energies'
MEL_PATH = DATA_PATH / 'mels'
PITCH_PATH = DATA_PATH / 'pitches'
WAV_PATH = DATA_PATH / 'LJSpeech-1.1/wavs'





def save_pitch():
    PITCH_PATH.mkdir(exist_ok=True, parents=True)

    wavs = []
    for file in WAV_PATH.iterdir():
        wavs.append(file.name)
    wavs.sort()

    min_pitch = np.inf
    max_pitch = -np.inf
    for i, wav_name in tqdm(enumerate(wavs), total=len(wavs)):
        mel = np.load(MEL_PATH / ("ljspeech-mel-%05d.npy" % (i + 1)))

        audio, sr = torchaudio.load(str(WAV_PATH / wav_name))
        audio = audio.double().numpy().sum(axis=0)

        frame_period = (audio.shape[0] / sr * 1000) / mel.shape[0]
        _f0, t = pw.dio(audio, sr, frame_period=frame_period)
        f0 = pw.stonemask(audio, _f0, t, sr)[: mel.shape[0]]
        nonzeros = np.nonzero(f0)
        x = np.arange(f0.shape[0])[nonzeros]
        values = (f0[nonzeros][0], f0[nonzeros][-1])
        f = interpolate.interp1d(x, f0[nonzeros], bounds_error=False, fill_value=values)
        new_f0 = f(np.arange(f0.shape[0]))

        np.save(PITCH_PATH / ("ljspeech-pitch-%05d.npy" % (i + 1)), new_f0)

        min_pitch = min(min_pitch, new_f0.min())
        max_pitch = max(max_pitch, new_f0.max())
    print(f"min_pitch={min_pitch}\tmax_pitch={max_pitch}")

if __name__ == '__main__':
    save_pitch()