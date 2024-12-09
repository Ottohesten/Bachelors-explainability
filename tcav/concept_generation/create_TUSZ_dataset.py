import os
import sys
from glob import glob
from tqdm import tqdm
from utils import process_TUSZ_edf
import numpy as np
import random


def process_TUEV_dir(dir_path, save_dir):
    """
    This function takes the path to the TUEV directory and processes the files in the directory.
    It adds annotations to the raw edf files, and saves them in the desired directory
    
    """
    edf_files = glob(f"{dir_path}/*/*/*/*.edf")
    print(edf_files[:10])
    print(f"Total files: {len(edf_files)}")
    # shuffle the files with seed
    random.seed(42)
    random.shuffle(edf_files)

    # get the first 1000 files
    # edf_files = edf_files[:1000]

    # get the combined file size of the subset of files
    # total_size = sum([os.path.getsize(file) for file in edf_files])
    # print(f"Total size: {total_size / 1e9} GB")

    # split into eval and train set

    split = 0.8
    split_idx = int(len(edf_files) * split)
    train_files = edf_files[:split_idx]
    eval_files = edf_files[split_idx:]


    # for edf_file in tqdm(edf_files):
    #     # process_TUSZ_edf(edf_file, save_dir, verbose=True)
    #     try:
    #         process_TUSZ_edf(edf_file, save_dir)
    #     except Exception as e:
    #         print(f"File: {edf_file}.\tError: {e}")
    #         continue

    for edf_file in tqdm(train_files):
        try:
            process_TUSZ_edf(edf_file, save_dir + "train/")
        except Exception as e:
            print(f"File: {edf_file}.\tError: {e}")
            continue
    
    for edf_file in tqdm(eval_files):
        try:
            process_TUSZ_edf(edf_file, save_dir + "eval/")
        except Exception as e:
            print(f"File: {edf_file}.\tError: {e}")
            continue


if __name__ == "__main__":
    TUEV_dir_path_train = "/nobackup/tsal-tmp/tuh_eeg_seizure/v2.0.0/edf/train/"
    # TUEV_dir_path_eval = "/nobackup/tsal-tmp/tuh_eeg_seizure/v2.0.0/edf/eval/"
    save_dir = "/scratch/s194101/TUSZ_processed_whole_dataset_split/"
    # save_dir = "/scratch/s194101/TUSZ_processed_eval/"

    process_TUEV_dir(TUEV_dir_path_train, save_dir)
    # process_TUEV_dir(TUEV_dir_path_eval, save_dir)