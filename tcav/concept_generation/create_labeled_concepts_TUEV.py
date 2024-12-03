import mne
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import pickle
import torch

def read_tuev_edf(file_path, process=True, notch_filter=False):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    if process:
        channel_map = {
            'EEG C3-REF': 'C3', 'EEG P4-REF': 'P4', 'EEG T5-REF': 'T5', 'EEG F8-REF': 'F8', 'EEG F7-REF': 'F7',
            'EEG C4-REF': 'C4', 'EEG PZ-REF': 'Pz', 'EEG FP2-REF': 'Fp2', 'EEG F4-REF': 'F4', 'EEG F3-REF': 'F3',
            'EEG T6-REF': 'T6', 'EEG CZ-REF': 'Cz', 'EEG O2-REF': 'O2', 'EEG O1-REF': 'O1', 'EEG T2-REF': 'F10',
            'EEG T1-REF': 'F9', 'EEG T4-REF': 'T4', 'EEG P3-REF': 'P3', 'EEG FZ-REF': 'Fz', 'EEG T3-REF': 'T3',
            'EEG FP1-REF': 'Fp1', 'EEG C4-LE': 'C4', 'EEG P3-LE': 'P3', 'EEG FZ-LE': 'Fz', 'EEG F3-LE': 'F3',
            'EEG FP1-LE': 'Fp1', 'EEG T6-LE': 'T6', 'EEG CZ-LE': 'Cz', 'EEG F8-LE': 'F8', 'EEG O1-LE': 'O1',
            'EEG PZ-LE': 'Pz', 'EEG C3-LE': 'C3', 'EEG FP2-LE': 'Fp2', 'EEG O2-LE': 'O2', 'EEG F7-LE': 'F7',
            'EEG T1-LE': 'T9', 'EEG T2-LE': 'F10', 'EEG P4-LE': 'P4', 
        }

    #     channel_map = {
    #     'EEG C3-REF': 'C3', 'EEG P4-REF': 'P4', 'EEG T5-REF': 'T5', 'EEG F8-REF': 'F8', 'EEG F7-REF': 'F7',
    #     'EEG C4-REF': 'C4', 'EEG PZ-REF': 'Pz', 'EEG FP2-REF': 'Fp2', 'EEG F4-REF': 'F4', 'EEG F3-REF': 'F3',
    #     'EEG T6-REF': 'T6', 'EEG CZ-REF': 'Cz', 'EEG O2-REF': 'O2', 'EEG O1-REF': 'O1', 'EEG T2-REF': 'F10',
    #     'EEG T4-REF': 'T4', 'EEG P3-REF': 'P3', 'EEG FZ-REF': 'Fz', 'EEG T3-REF': 'T3',
    #     'EEG FP1-REF': 'Fp1', 'EEG C4-LE': 'C4', 'EEG P3-LE': 'P3', 'EEG FZ-LE': 'Fz', 'EEG F3-LE': 'F3',
    #     'EEG FP1-LE': 'Fp1', 'EEG T6-LE': 'T6', 'EEG CZ-LE': 'Cz', 'EEG F8-LE': 'F8', 'EEG O1-LE': 'O1',
    #     'EEG PZ-LE': 'Pz', 'EEG C3-LE': 'C3', 'EEG FP2-LE': 'Fp2', 'EEG O2-LE': 'O2', 'EEG F7-LE': 'F7',
    #     'EEG T1-LE': 'T9', 'EEG P4-LE': 'P4', 
    # }

        channels_to_remove = ["F10", "F9"]

        channel_map_sub = {k: v for k, v in channel_map.items() if k in raw.ch_names and v not in channels_to_remove}
        
        # standardize the raw data
        mne.datasets.eegbci.standardize(raw)

        # rename the channels that are in the channel_map_sub
        raw = raw.rename_channels(channel_map_sub)

        rename_map = {'T3': 'T7', 'T4': 'T8', 'P7': 'T5', 'P8': 'T6'}
        rename_map = {k: v for k, v in rename_map.items() if k in raw.ch_names}
        raw.rename_channels(rename_map)

        # need to rename them in channel_map_sub as well
        final_list = list(channel_map_sub.values())
        for k, v in rename_map.items():
            final_list[final_list.index(k)] = v

        # make standard 10-20 montage
        montage = mne.channels.make_standard_montage('standard_1020')
        # set the montage
        raw = raw.set_montage(montage, on_missing='ignore', verbose=False)
        
        # this is where we actually make it so we are only left with the channels we want
        # raw = raw.pick(list(channel_map_sub.values()))
        raw = raw.pick(final_list)

        # Set the average reference for the raw data
        raw = raw.set_eeg_reference(ref_channels='average', projection=True, verbose=False)

        # where we actually apply the projection from above
        raw.apply_proj(verbose=False)

        # resample the data to 256 Hz
        raw = raw.resample(256)

        if notch_filter:
            raw = raw.notch_filter(60.0, fir_design='firwin', verbose=False)


    return raw

def get_window(raw, annotation, window_size):
    """
    given a raw object and an annotation, return a window of the data
    """

    onset, duration, description, _ = annotation.values()

    start = int(onset * raw.info['sfreq'])
    
    # end is the start plus the window size
    end = start + int(window_size * raw.info['sfreq'])

    # if the end is longer than the data we can not procede, otherwise we will get an error
    if end > raw.get_data().shape[1]:
        return None

    window = raw.copy().get_data()[:, start:end]

    return window




def create_tuev_concepts(edf_files, save_dir, window_length):
    """
    Go through all the edf files in the list and create the concepts and save them in the save_dir as pickle files with appropriate names
    
    """
    for file_path in tqdm(edf_files):
        raw = read_tuev_edf(file_path)
        annotations = raw.annotations

        for idx, annotation in enumerate(annotations):
            onset, duration, description, _ = annotation.values()
            window = get_window(raw, annotation, window_length)
            if window is None:
                continue

            # check if the dir already exists, otherwise create it
            concept_dir = f"{save_dir}/{description}"
            if not os.path.exists(concept_dir):
                os.makedirs(concept_dir)
            
            save_path = f"{concept_dir}/{os.path.basename(file_path).split('.')[0]}_{idx}.pkl"
            # print(save_path)
            with open(save_path, 'wb') as f:
                # save as torch tensor
                pickle.dump(torch.tensor(window, dtype=torch.float32), f, protocol=pickle.HIGHEST_PROTOCOL)








if __name__ == "__main__":
    TUEV_dir_path = "/scratch/s194101/TUEV_processed"
    save_dir = "/scratch/s194101/concepts/TUEV"

    edf_files = glob(f"{TUEV_dir_path}/*/*.edf")
    print(f"Number of edf files: {len(edf_files)}")
    # edf_files = edf_files[:10]
    # print(edf_files[:10])

    create_tuev_concepts(edf_files, save_dir, 60.0)


    