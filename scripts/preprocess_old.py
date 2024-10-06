from typing import Tuple
from jsonargparse import CLI
import h5py
import glob
import random
import numpy as np
import os
import mne
from braindecode.preprocessing.preprocess import _preprocess as preprocess_fn
from braindecode.preprocessing.windowers import _create_fixed_length_windows as fixed_length_windows
from braindecode.preprocessing import Preprocessor
from eegatscale.BENDR.deep1010 import interpolate_nearest, to_1020, to_deep1010, simple_filter
from braindecode.datasets.base import BaseDataset
import tqdm
from abc import abstractmethod, ABC

mne.set_log_level("CRITICAL")


class PreprocessPipeline(ABC):
    @abstractmethod
    def __call__(self, raw: mne.io.Raw) -> mne.io.Raw:
        """overwrite"""

class BendrPreprocessPipeline(PreprocessPipeline):
    def __init__(self, data_min: float, data_max: float, sfreq: float) -> None:
        self.preprocessors = [
            Preprocessor(interpolate_nearest, sfreq=sfreq, apply_on_array=False),
            Preprocessor(simple_filter, apply_on_array=False),
            Preprocessor(to_deep1010, data_min=data_min, data_max=data_max, apply_on_array=False),
            Preprocessor(to_1020, apply_on_array=False),
        ]

    def __call__(self, raw: mne.io.Raw) -> mne.io.Raw:
        raw.load_data()
        dataset = BaseDataset(raw)
        processed_dataset = preprocess_fn(
            dataset, ds_index=None, preprocessors=self.preprocessors, save_dir=None
        )
        return processed_dataset.raw


class Dataset(ABC):
    def __init__(self, tmin: float, tlen: float) -> None:
        self.tmin = tmin
        self.tlen = tlen

    @abstractmethod
    def file_reader(self, file: str) -> mne.io.Raw:
        """Overwrite"""

    @abstractmethod
    def window_fn(self, dataset: BaseDataset) -> Tuple[np.ndarray, np.ndarray]:
        """Overwrite"""

    def __len__(self):
        return len(self.files)

    def tmax(self, sfreq: float) -> float:
        return self.tmin + self.tlen - 1 / sfreq

class Sleep(Dataset):
    name = "sleep"
    channel_mapping = {'Fpz-Cz': "FPZ", 'Pz-Oz': "PZ", "horizontal": "HEOGL"}
    remove_channels = ["oro-nasal", 'submental', 'rectal', 'Event marker']

    def __init__(self, path: str, tmin: float, tlen: float) -> None:
        super().__init__(tmin, tlen)
        self.files = glob.glob(os.path.join(path, "**/*-PSG.edf"), recursive=True)
        self.annotations = glob.glob(os.path.join(path, "**/*-Hypnogram.edf"), recursive=True)
        self.mapping = { }
        for f in self.files:
            for an in self.annotations:
                if f.split("/")[-1][:6] == an.split("/")[-1][:6]:
                    self.mapping[f] = an

    def file_reader(self, file):
        raw = mne.io.read_raw_edf(file, stim_channel="Event marker", infer_types=True)
        annotation = mne.read_annotations(self.mapping[file])
        raw.set_annotations(annotations=annotation, emit_warning=False)
        raw.rename_channels(self.channel_mapping)
        raw.drop_channels(self.remove_channels)
        return raw

    def window_fn(self, raw):
        events = mne.events_from_annotations(raw)
        event_dict = {
            'Sleep stage W': 0, 
            'Sleep stage 1': 1, 
            'Sleep stage 2': 2, 
            'Sleep stage 3': 3, 
            'Sleep stage 4': 3, 
            'Sleep stage R': 4,
        }
        epoch = mne.Epochs(
            raw, events[0], event_id=event_dict, tmin=self.tmin, tmax=self.tmax(raw.info["sfreq"]), baseline=None
        )
        return epoch.get_data(), epoch.events[:,-1] - min(event_dict.values())


class MMI(Dataset):
    name = "mmi"
    exclude_people = ["S088", "S090", "S092", "S100"]
    exclude_sessions = [
        f"R{str(session_number).zfill(2)}" for session_number in range(1, 15) if session_number not in [6, 10, 14]
    ]

    def __init__(self, path: str, tmin: float, tlen: float) -> None:
        super().__init__(tmin, tlen)
        self.path = path
        self.files = glob.glob(os.path.join(self.path, "**/*.edf"), recursive=True)
        self.files = [file for file in self.files if not any(person in file for person in self.exclude_people)]

    def file_reader(self, file):
        raw = mne.io.read_raw_edf(file)
        if any(session in raw.filenames[0].split("/")[-1] for session in self.exclude_sessions):
            return None
        return raw
    
    def window_fn(self, raw):
        events = mne.events_from_annotations(raw)
        event_dict = {'T1': events[1]["T1"], "T2": events[1]["T2"]}
        epoch = mne.Epochs(
            raw, events[0], event_id=event_dict, tmin=self.tmin, tmax=self.tmax(raw.info["sfreq"]), baseline=None
        )
        return epoch.get_data(), epoch.events[:,-1] - min(event_dict.values())


class BCI(Dataset):
    name = "bci"
    channel_mapping = {
        'EEG-Fz': "Fz", 
        'EEG-0': "FC3", 
        'EEG-1': "FC1", 
        'EEG-2': "FCz", 
        'EEG-3': "FC2", 
        'EEG-4': "FC4", 
        'EEG-5': "C5", 
        'EEG-C3': "C3", 
        'EEG-6': "C1", 
        'EEG-Cz': "Cz", 
        'EEG-7': "C2", 
        'EEG-C4': "C4", 
        'EEG-8': "C6", 
        'EEG-9': "CP3", 
        'EEG-10': "CP1", 
        'EEG-11': "CPz", 
        'EEG-12': "CP2", 
        'EEG-13': "CP4", 
        'EEG-14': "P1", 
        'EEG-Pz': "Pz", 
        'EEG-15': "P2", 
        'EEG-16': "POz",
    }

    def __init__(self, path: str, tmin: float, tlen: float) -> None:
        super().__init__(tmin, tlen)
        self.path = path
        self.files = glob.glob(os.path.join(self.path, "**/*T.gdf"), recursive=True)

    def file_reader(self, file):
        raw = mne.io.read_raw_gdf(file, eog=["EOG-left", "EOG-central", "EOG-right"])
        raw.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
        raw.rename_channels(self.channel_mapping)
        return raw

    def window_fn(self, raw):
        events=mne.events_from_annotations(raw)
        event_dict = {
            "769": events[1]["769"], "770": events[1]["770"], "771": events[1]["771"], "772": events[1]["772"]
        }
        epoch=mne.Epochs(
            raw, events[0], event_id=event_dict, tmin=self.tmin, tmax=self.tmax(raw.info["sfreq"]), baseline=None
        )
        return epoch.get_data(), epoch.events[:,-1] - min(event_dict.values())


class TUH(Dataset):
    name = "tuh"
    def __init__(self, path: str, tmin: float, tlen: float) -> None:
        super().__init__(tmin, tlen)
        self.path = path
        self.files = glob.glob(os.path.join(self.path, "**/*.edf"), recursive=True)

    def file_reader(self, file):
        raw = mne.io.read_raw_edf(file)        
        duration = len(raw) / raw.info["sfreq"]
        if duration < self.tlen:
            return None
        return raw
    
    def window_fn(self, raw):
        sfreq = raw.info["sfreq"]
        dataset = BaseDataset(raw)

        windows = fixed_length_windows(
            dataset,
            start_offset_samples=int(self.tmin),
            stop_offset_samples=None,
            window_size_samples=int(sfreq * self.tlen),
            window_stride_samples=int(sfreq * self.tlen),
            drop_last_window=True,
        )
        windows = np.stack([w[0] for w in windows], 0)
        return windows, -1 * np.ones((windows.shape[0],))


def preprocess_datasets(
    dataset: Dataset,
    pipeline:  ,
    out_path: str,
    num_recordings: int = -1,
):

    file_cond = num_recordings != -1 and num_recordings < len(dataset)
    if file_cond:
        # randomly sample
        choices = random.choices(range(len(dataset)), k=num_recordings)
        dataset.files = [f for i, f in enumerate(dataset.files) if i in choices]
    
    counter = 0
    print(f"preprocessing {len(dataset)} files")
    for idx, f in enumerate(tqdm.tqdm(dataset.files)):
        raw = dataset.file_reader(f)
        if raw is None:
            continue
    
        processed_raw = pipeline(raw)
        data, labels = dataset.window_fn(processed_raw)
        n_elem, n_channels, sequence_length = data.shape

        if idx == 0:
            extension = str(num_recordings) if file_cond else "all"
            file = h5py.File(f"{out_path}/{dataset.name}_preprocessed_{extension}.hdf5", "w")
            #file.attrs["files"] = dataset.files
            file.create_dataset(
                "data", shape=(1000, n_channels, sequence_length), maxshape=(None, n_channels, sequence_length)
            )
            file.create_dataset("labels", shape=(1000,), maxshape=(None,))

        current_size = file["data"].shape[0]

        if counter + n_elem >= current_size:  # grow array as needed
            file["data"].resize((2*current_size, n_channels, sequence_length))
            file["labels"].resize((2*current_size,))
            print(f"growing array from {current_size} to {2*current_size}")
        
        file["data"][counter:counter+n_elem] = data
        file["labels"][counter:counter+n_elem] = labels
        counter += n_elem

    # cut away if we grew the array too large
    print(f"resizing array from {current_size} to {counter}")
    file["data"].resize((counter, n_channels, sequence_length))
    file["labels"].resize((counter,))

    file.close()


if __name__ == "__main__":
    mne.set_log_level("CRITICAL")
    CLI(preprocess_datasets)