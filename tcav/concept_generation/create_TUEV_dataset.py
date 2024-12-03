import os
import sys
from glob import glob
from tqdm import tqdm
from utils import process_TUEV_edf


def process_TUEV_dir(dir_path, save_dir):
    """
    This function takes the path to the TUEV directory and processes the files in the directory.
    It adds annotations to the raw edf files, and saves them in the desired directory
    
    """
    edf_files = glob(f"{dir_path}/*/*.edf")
    print(f"Total files: {len(edf_files)}")

    for edf_file in tqdm(edf_files):
    # for edf_file in edf_files:
        # print(f"Processing {edf_file}")
        # process_TUEV_edf(edf_file, save_dir, verbose=True)
        try:
            process_TUEV_edf(edf_file, save_dir)
        except Exception as e:
            print(f"File: {edf_file}.\tError: {e}")
            continue


if __name__ == "__main__":
    TUEV_dir_path = "/nobackup/tsal-tmp/tuh_eeg_events/v2.0.0/edf/train/"
    save_dir = "/scratch/s194101/TUEV_processed/"

    print("processing TUEV directory")
    process_TUEV_dir(TUEV_dir_path, save_dir)