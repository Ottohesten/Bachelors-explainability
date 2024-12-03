import mne
from mne.datasets import fetch_fsaverage
import numpy as np
from pathlib import Path
import os

from typing import Dict, List, Tuple, Union

def get_fsaverage(verbose = False):
    """Returns the fsaverage files.
    Parameters
    ----------
    verbose : bool
        Whether to print the progress or not.
    Returns
    ----------
    subjects_dir : str
        The subjects directory.
    subject : str
        The subject.
    trans : str
        The transformation.
    src_path : str
        The source path.
    bem_path : str
        The bem path.
    """
    # Download fsaverage files
    fs_dir = Path(fetch_fsaverage(verbose=False))
    subjects_dir = os.path.dirname(fs_dir)

    # The files live in:
    subject = 'fsaverage'
    trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
    src_path = fs_dir / 'bem' / 'fsaverage-ico-5-src.fif'
    bem_path = fs_dir / 'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif'

    return subjects_dir, subject, trans, src_path, bem_path


def get_raw(edf_file_path: Path, filter: bool = True,
            high_pass = 1.0, low_pass = 4.0, notch = 60, resample = 256, proj=True) -> mne.io.Raw:
    """Reads an edf file and returns a raw object.
    Parameters
    ----------
    edf_file_path : str
        Path to the edf file.
    filter : bool
        Whether to filter the data or not.
    Returns
    -------
    raw : mne.io.Raw
        The raw object.
    """
    raw = mne.io.read_raw_edf(edf_file_path, verbose=False, preload=True)
    mne.datasets.eegbci.standardize(raw)  # Set channel names
    montage = mne.channels.make_standard_montage('standard_1020')

    new_names = dict(
        (ch_name,
        ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp'))
        for ch_name in raw.ch_names)
    raw.rename_channels(new_names)
    
    raw = raw.set_eeg_reference(ref_channels='average', projection=True, verbose = False)
    raw = raw.set_montage(montage); # Set montage
    if proj:
        raw.apply_proj(verbose = False)

    if filter:
        if resample:
            raw = raw.resample(resample)
        if low_pass and high_pass:
            raw = raw.filter(high_pass, low_pass, verbose = False)
        if notch:
            raw = raw.notch_filter(notch, verbose = False, filter_length='auto')
            # raw = raw.notch_filter(notch, verbose = False, filter_length='3s')

    return raw