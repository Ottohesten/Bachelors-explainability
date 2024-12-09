import mne
import numpy as np

import os

from glob import glob
from tqdm import tqdm
from utils import read_TUH_edf
import pickle
import torch




def process_file(file_path, n_samples, sample_length, save_dir):
    channel_order = [
                'Fp1', 'Fp2',
        'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
                 'O1', 'O2'
    ]
    
    raw = read_TUH_edf(file_path)
    try:
        raw.pick(channel_order)
        raw.reorder_channels(channel_order)
    except:
        print("Error in picking and reordering channels")
        return

    sample_freq = raw.info['sfreq']

    for i in range(n_samples):
        # because the sample is 60 seconds long, we need to take a random start time that is at least 60 seconds from the end
        # start_time = np.random.randint(0, raw.times[-1] - sample_length)
        # sample = raw.copy().crop(tmin=start_time, tmax=start_time + sample_length)

        start = np.random.randint(0, raw.n_times - sample_length * sample_freq)
        length = int(sample_length * sample_freq)

        sample = raw.copy().get_data()[:, start:start + length]

        x = torch.from_numpy(sample)

        # check if channels is 19, if not skip
        # print(x.shape)
        if x.shape[0] != 19:
            print(f"Skipping {file_path} because of shape {x.shape}")
            continue

        # save the sample as pickle file
        save_file = os.path.join(save_dir, os.path.basename(file_path).replace(".edf", f"_sample_{i}.pkl"))
        with open(save_file, 'wb') as f:
            pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)

    
if __name__ == "__main__":
    dir_path = "/scratch/agjma/tuh_selected/"

    edf_files = glob(os.path.join(dir_path, "*.edf"))

    print(f"Found {len(edf_files)} edf files")

    save_dir = "/scratch/s194101/random_tuh_selected_samples_60/"
    # check ir directory exists, create if not
    os.makedirs(save_dir, exist_ok=True)
    for file_path in tqdm(edf_files):
        process_file(file_path, n_samples=2, sample_length=60, save_dir=save_dir)