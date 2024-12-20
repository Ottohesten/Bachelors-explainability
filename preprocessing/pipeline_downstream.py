import mne
import logging
from pathlib import Path
import logging
from preprocessing.utils import split_raw, get_unannotated_raw, split_raw_annotations
from preprocessing.methods import PreprocessMethods

# Import typing
from typing import Tuple, List, Optional

from preprocessing.pipeline import BasePipeline

class DownstreamPipeline(BasePipeline):
    def __init__(self, descriptions: List[str], tmin: float = -0.5, tlen: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.descriptions = descriptions
        self.description_map = {label: i for i, label in enumerate(descriptions)}
        self.tmin, self.tlen = tmin, tlen
        
    def __call__(self, src_paths: List[str]) -> Tuple[List[mne.io.Raw], List[Tuple[float, float]], List[int]]:
        return self.run(src_paths)
    
    def run(self, src_paths):
        logging.debug("Loading EDF files...")
        src_paths = [Path(src_path) for src_path in src_paths]

        raws = []
        
        logging.debug("Splitting raws...")   
        for i, src_path in enumerate(src_paths):
            try:
                raw_orig = mne.io.read_raw_edf(src_path, preload=True, verbose=False)
                
                # Rename channels with channel_rename
                if self.channels_rename is not None:
                    raw_orig.rename_channels(self.channels_rename)
                    logging.info(f"File: {src_paths[i].stem}.\tRenamed channels: {self.channels_rename}.")
                           
                self._to_standard_names(raw_orig)
                drop_chs = self._set_montage(raw_orig)
                logging.info(f"File: {src_paths[i].stem}.\tDropped {len(drop_chs)} channels when setting montage: {drop_chs}.")
            except Exception as e:
                logging.error(f"Dropping file: {src_path.stem}.\tError: {e}")
                continue

            raws.append(raw_orig)
            
        total_windows = len(raws)
        logging.debug(f"Total files: {total_windows}")
                       
        for i, raw in enumerate(raws):
            filename = src_paths[i]
            
            try:
                raws[i] = self.run_single(raw, filename)
            except Exception as e:
                logging.error(f"File: {filename}.\tError: {e}")
                raws[i] = None
                
            # Log progress every N windows
            if (i + 1) % 10 == 0:
                logging.debug(f"Processed {i + 1}/{total_windows} files.")
     
        # Update raws
        src_paths = [src_path for i, src_path in enumerate(src_paths) if raws[i] is not None]
        raws = [raw for raw in raws if raw is not None]
     
        raw_windows = []
        times = []
        indices = []
        descriptions = []
        sessions_labels = []
        file_paths = []
        export = False
        # test_data_dir = "/home/s194101/Bachelors-explainability/raws_test"
        test_data_dir = "/scratch/s194101/data/preprocessed_downstream/mmidb_noica_5.0_titans_windows_no_anno"
        
        for i, raw in enumerate(raws):
            windows, time_slices, descri = split_raw_annotations(raw, labels = self.descriptions, tmin=self.tmin,
                                                                 tlen=self.tlen, verbose=False)
            raw_windows.extend(windows)
            times.extend(time_slices)
            descriptions.extend(descri)
            indices.extend([i] * len(windows))
            logging.debug(f"raw filename: {raw.filenames[0]}")
            # sessions_labels.extend(int(raw.filenames[0].split("/")[-2][1:]) for _ in range(len(windows))) # the session label is just the subject which is the parent directory name number
            sessions_labels.extend(int(0) for _ in range(len(windows))) # the session label is just the subject which is the parent directory name number
            file_path = raw.filenames[0]
            # filename = raw.filenames[0].split("/")[-1].split(".")[0]
            # run = raw.filenames[0].split("/")[-1].split(".")[0].split("R")[-1]
            file_paths.extend([file_path] * len(windows))
            if export:
                for i, (window, description) in enumerate(zip(windows, descri)):
                    # window.save(f"{test_data_dir}/{file_path.split('/')[-1].split('.')[0]}_{i}.edf", overwrite=True) # using save
                    # window.export(f"{test_data_dir}/{file_path.split('/')[-1].split('.')[0]}_{i}_no_anno.edf") # using export
                    window.export(f"{test_data_dir}/{description}/{file_path.split('/')[-1].split('.')[0]}_{i}.edf") # using export
                    

            
        labels = [self.description_map[description] for description in descriptions]

        # session is just the subject number what is the parent directory name number

        assert len(raw_windows) == len(times) == len(indices) == len(labels) == len(sessions_labels) == len(file_paths), "Lengths of raw_windows, times, indices, labels, sessions_labels and file_paths should be equal."
        
        return raw_windows, times, indices, labels, sessions_labels, file_paths
        # return raw_windows, times, indices, labels
    
    def run_single(self, raw, filename) -> Optional[mne.io.Raw]:
        window_info_str = f"File: {filename}."
        
        self._remove_line_noise(raw)
        
        raw_unannotated = raw #get_unannotated_raw(raw, resting_state=['T0'])
        bad_chs = self._find_bad_channels(raw_unannotated)
        logging.info(f"{window_info_str}\tFound {len(bad_chs)} bad channels: {bad_chs}.")
        
        # Drop bad channels
        raw.drop_channels(bad_chs)
        
        self._filter(raw)    
        self._average_reference(raw)
                
        if self.do_ica:
            excluded_idxs, labels, y_proba = self._ica_clean(raw)
            logging.info(f"{window_info_str}\tExcluding {len(excluded_idxs)} components: {excluded_idxs}.")
            logging.info(f"{window_info_str}\tLabels: {labels}.")
            logging.info(f"{window_info_str}\tProbabilities: {[round(prob, 2) for prob in y_proba]}.")
            
            raw_unannotated = raw #get_unannotated_raw(raw, resting_state=['T0'])
            bad_chs = self._find_bad_channels(raw_unannotated)
            logging.info(f"{window_info_str}\tFound {len(bad_chs)} bad channels: {bad_chs}.")
            
            # Drop bad channels
            raw.drop_channels(bad_chs)

        missing_chs = self._interpolate_missing(raw)
        logging.info(f"{window_info_str}\tIntepolating {len(missing_chs)} channels: {missing_chs}.")
        
        extra_chs = self._drop_extra_and_reorder(raw)
        logging.info(f"{window_info_str}\tRemoving {len(extra_chs)} extra channels: {extra_chs}.")
        
        self._interpolate_nearest(raw)        
        return raw
    
    def _find_bad_channels(self, raw: mne.io.Raw, drop=True):       
        return PreprocessMethods.find_bad_channels(raw, ransac = self.ransac, drop = False)
    
class DownstreamPipelineBENDR(DownstreamPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _zero_missing(self, raw: mne.io.Raw):
        return PreprocessMethods.zero_missing(raw, self.chs, self.montage)
    
    def run_single(self, raw, filename) -> Optional[mne.io.Raw]:
        window_info_str = f"File: {filename}."
        
        missing_chs = self._zero_missing(raw)
        logging.info(f"{window_info_str}\Zeroing {len(missing_chs)} channels: {missing_chs}.")
        
        extra_chs = self._drop_extra_and_reorder(raw)
        logging.info(f"{window_info_str}\tRemoving {len(extra_chs)} extra channels: {extra_chs}.")
        
        self._interpolate_nearest(raw)        
        return raw
    
    def _find_bad_channels(self, raw: mne.io.Raw, drop=True):       
        return PreprocessMethods.find_bad_channels(raw, ransac = self.ransac, drop = False)