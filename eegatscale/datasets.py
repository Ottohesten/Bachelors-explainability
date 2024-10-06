import contextlib
import glob
import io
import os
from typing import Any, Callable, Dict, Iterable, List

import h5py
import numpy as np
import pandas as pd
from braindecode.datasets import tuh
from braindecode.datasets.base import BaseDataset
from braindecode.preprocessing import Preprocessor
from braindecode.preprocessing.preprocess import _preprocess
from torch.utils.data import Dataset


def _parse_description_from_file_path(file_path: str) -> Dict[str, Any]:
    # stackoverflow.com/questions/3167154/how-to-split-a-dos-path-into-its-components-in-python  # noqa
    file_path = os.path.normpath(file_path)
    tokens = file_path.split(os.sep)
    # expect file paths as tuh_eeg/version/file_type/reference/data_split/subject/recording session/file
    # e.g.                 tuh_eeg/v1.1.0/edf/01_tcp_ar/027/00002729/s001_2006_04_12/00002729_s001.edf

    # indexing etc. changes made in order to fit to /nobackup/tsal-tmp where the file paths are like:
    # tuh_eeg/v2.0.0/edf/000/aaaaaaan/s001_2002_09_03/02_tcp_le/aaaaaaan_s001_t001.edf
    version = tokens[-7]
    year, month, day = tokens[-3].split("_")[1:]
    subject_id = tokens[-4]
    session = tokens[-3].split("_")[0]
    segment = tokens[-1].split("_")[-1].split(".")[-2]

    return {
        "path": file_path,
        "version": version,
        "year": int(year),
        "month": int(month),
        "day": int(day),
        "subject": str(subject_id),
        "session": int(session[1:]),
        "segment": int(segment[1:]),
    }


class TUH(tuh.TUH):
    """Plug in replacement for the TUH class from braindecode with the added functionality:
    * Online loading of data
    * Custom path for data
    * Online preprocessing functionality

    All arguments are the exact same as the base class except for the `offline`, `path_layout` and `online_preprocessor`
    arguments. Initializing the class on the full dataset takes around 35 min in offline mode whereas it takes around
    2 min in online mode.

    Args:
        path: base path where dataset is stored
        recording_ids: optional list of indexes for the recordings to get
        target_name: either `age` or `gender`
        preload: if data should be loaded immidiatly when it is read from disk or when the first operation is done
        add_physian_reports: if physian reports should be added to samples
        n_jobs: number of jobs used for loading data in offline setting
        offline: if data should be loaded offline or online
        path_layout: a callable function the describes how the data is organised locally
        preprocess: Iterable of preprocessor functions that are combatible with `preprocess` from braindecode.
            Only used when running in online mode.
    """

    def __init__(
        self,
        path: str,
        recording_ids: None | List[int] = None,
        target_name: None | str = None,
        preload: bool = False,
        add_physician_reports: bool = False,
        n_jobs: int = 1,  # if offline=False, then this has no effect
        offline: bool = True,
        path_layout: None | Callable = _parse_description_from_file_path,
        preprocess: None | Iterable[Preprocessor] = None, # if offline=True, then this has no effect
    ):
        if path_layout is not None:
            if callable(path_layout):
                tuh._parse_description_from_file_path = _parse_description_from_file_path
            else:
                raise ValueError(f"Expected argument `path_layout` to be a callable function, but got {path_layout}")

        self.offline = offline
        if self.offline:
            super().__init__(path, recording_ids, target_name, preload, add_physician_reports, n_jobs)
        else:
            # save for later
            self._target_name = target_name
            self._preload = preload
            self._add_physician_reports = add_physician_reports
            self._preprocess = preprocess

            file_paths = glob.glob(os.path.join(path, "**/*.edf"), recursive=True)
            descriptions = tuh._create_chronological_description(file_paths)
            if recording_ids is not None:
                descriptions = descriptions[recording_ids]
            self._description = descriptions
            self._dataset = None

    @property
    def description(self) -> pd.DataFrame:
        """Get dataset description."""
        if self.offline:
            return super().description
        else:
            df = pd.DataFrame(self._description).T
            df.reset_index(inplace=True, drop=True)
            return df

    @property
    def datasets(self):
        if self.offline:
            return self._dataset
        else:
            return OnlineDataset(self)

    @datasets.setter
    def datasets(self, value) -> None:
        if self.offline:
            self._dataset = value
        else:
            raise ValueError("When `offline=False` you should not set the `datasets` attribute")

    def __len__(self) -> int:
        if self.offline:
            return super().__len__()
        else:
            raise ValueError("Cannot get the length of when `offline=True` because that requires loading all data")


class OnlineDataset(Dataset):
    """Mock version of the dataset classes from TUH."""

    def __init__(self, tuh_instance: TUH) -> None:
        self.tuh_instance = tuh_instance

    def __getitem__(self, index: int) -> BaseDataset:
        with contextlib.redirect_stdout(io.StringIO()):  # removes printing done by mne
            dataset = self.tuh_instance._create_dataset(
                description=self.tuh_instance._description[index],
                target_name=self.tuh_instance._target_name,
                preload=self.tuh_instance._preload,
                add_physician_reports=self.tuh_instance._add_physician_reports,
            )
            if self.tuh_instance._preprocess is not None:
                dataset = _preprocess(ds=dataset, ds_index=None, preprocessors=self.tuh_instance._preprocess)
            return dataset

    def __len__(self) -> int:
        return self.tuh_instance._description.shape[-1]

    def __iter__(self) -> Iterable:
        for index in range(self.tuh_instance._description.shape[-1]):
            yield self.__getitem__(index)

class PreprocessedBENDRDataset(Dataset):
    """Loads in preprocessed data used in BENDR saved to HDF5 format"""

    def __init__(self, path: str) -> None:
        self.file = h5py.File(path, "r")
        self.data = self.file[list(self.file.keys())[0]]

    def __getitem__(self, index: int) -> np.ndarray:
        return self.data[index]
    
    def __len__(self) -> int:
        return self.data.shape[0]
    
    def __iter__(self) -> Iterable:
        for index in range(self.data.shape[0]):
            yield self.__getitem__(index)
    
    def __del__(self) -> None:
        self.file.close()
    
    def close(self) -> None:
        self.file.close()
    