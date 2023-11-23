from tqdm import tqdm
from pathlib import Path

import numpy as np

DATA_PATH = Path('data')
ENERGY_PATH = DATA_PATH / 'energies'
MEL_PATH = DATA_PATH / 'mels'

def save_energy():
    ENERGY_PATH.mkdir(exist_ok=True, parents=True)
    min_energy = np.inf
    max_energy = -np.inf
    for mel_path in tqdm(MEL_PATH.iterdir()):
        mel = np.load(mel_path)
        energy = np.linalg.norm(mel, axis=-1)
        np.save(ENERGY_PATH / mel_path.name.replace("mel", "energy"), energy)
        min_energy = min(min_energy, energy.min())
        max_energy = max(max_energy, energy.max())
    print(f"min_energy: {min_energy}\nmax_energy: {max_energy}")

if __name__ == '__main__':
    save_energy()
