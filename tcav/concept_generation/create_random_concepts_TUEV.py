import mne
import numpy as np

import os

from glob import glob
from tqdm import tqdm
from utils import read_TUH_edf
import pickle
import torch



def process_file(filepath, save_dir, n_samples=2, sample_length=60):
    channel_order = [
                'Fp1', 'Fp2',
        'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
                 'O1', 'O2'
    ]
    raw = read_TUH_edf(filepath)
    try:
        raw.pick(channel_order)
        raw.reorder_channels(channel_order)
    except:
        print("Error in picking and reordering channels")
        return

    sample_freq = raw.info['sfreq'] # 256
    for i in range(n_samples):
        start = np.random.randint(0, raw.n_times - sample_length * sample_freq)
        length = int(sample_length * sample_freq)


        sample = raw.copy().get_data()[:, start:start + length]

        x = torch.from_numpy(sample)

        if x.shape[0] != 19: # needs to be 19 channels
            print(f"Skipping {filepath} because of shape {x.shape}")
            continue

        # save as pickle file
        save_path = os.path.join(save_dir, os.path.basename(filepath).replace(".edf", f"_sample_{i}.pkl"))
        with open(save_path, 'wb') as f:
            pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    dir_path = "/nobackup/tsal-tmp/tuh_eeg_events/v2.0.0/edf/train/"

    edf_files = glob(f"{dir_path}/*/*.edf")

    print(f"Found {len(edf_files)} edf files")

    save_dir = "/scratch/s194101/concepts/tuev_random_concepts/"
    # check ir directory exists, create if not
    os.makedirs(save_dir, exist_ok=True)
    for file_path in tqdm(edf_files):
        process_file(file_path, n_samples=2, sample_length=60, save_dir=save_dir)
