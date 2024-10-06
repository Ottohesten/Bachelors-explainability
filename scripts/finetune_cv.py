from jsonargparse import CLI
import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from eegatscale.models import LinearHeadBENDR, FullBENDR
from eegatscale.transforms import StandardizeLabel
from typing import Optional

class H5PYDatasetLabeled(Dataset):
    def __init__(self, path: str, transform=None):
        if os.path.isdir(path):
            self.paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.hdf5')]
        else:
            raise ValueError("Path should be a directory")
        
        self.paths.sort()
        self.data_key = 'data'
        self.label_key = 'labels'
        self.lengths = [self._get_file_length(path) for path in self.paths]
        
        self.sessions = torch.concat([self._get_session(path) for path in self.paths])
        self.labels = torch.cat([self._get_labels(path) for path in self.paths])
        
        self.cumulative_lengths = self._compute_cumulative_lengths(self.lengths)
        
        if transform is not None:
            self.transform = transform

    def _get_file_length(self, path):
        with h5py.File(path, 'r') as file:
            return file[self.data_key].shape[0]
        
    def _get_session(self, path):
        with h5py.File(path, 'r') as file:
            return torch.from_numpy(file['sessions_labels'][:])
        
    def _get_labels(self, path):
        with h5py.File(path, 'r') as file:
            return torch.from_numpy(file[self.label_key][:])

    def _compute_cumulative_lengths(self, lengths):
        cumulative_lengths = [0]
        for length in lengths:
            cumulative_lengths.append(cumulative_lengths[-1] + length)
        return cumulative_lengths

    def __len__(self):
        return self.cumulative_lengths[-1]

    def _load_data(self, path, local_index):
        with h5py.File(path, 'r') as file:
            data = file[self.data_key][local_index]
            label = file[self.label_key][local_index]
            return torch.from_numpy(data), label

    def __getitem__(self, global_index: int):
        # If global_index is out of bounds, raise an error
        if global_index < 0 or global_index >= len(self):
            raise IndexError(f"Index {global_index} out of bounds for dataset of length {len(self)}")
        
        file_index = self._find_file_index(global_index)
        local_index = global_index - self.cumulative_lengths[file_index]
        data, label = self._load_data(self.paths[file_index], local_index)
        
        if hasattr(self, 'transform'):
            data, label = self.transform((data, label))
        
        return data, label

    def _find_file_index(self, global_index):
        # Binary search to find the right file index
        low, high = 0, len(self.cumulative_lengths) - 1
        while low < high:
            mid = (low + high) // 2
            if global_index < self.cumulative_lengths[mid + 1]:
                high = mid
            else:
                low = mid + 1
        return low
    
    def teardown(self):
        pass
    
def finetune_cv(dataset_path: str, encoder_path: Optional[str], name: str, n_splits: int = 10, batch_size: int = 16,
                num_workers: int = 4, num_epochs: int = 50, seed: int = 42, n_repeats: int = 5, n_device: int = 1,
                device: str = 'cuda', out_features: int = 2, freeze_encoder: bool = False):
    
    group_kfold = StratifiedGroupKFold(n_splits=n_splits)
    #group_kfold = StratifiedKFold(n_splits=n_splits)
    
    # Set all seeds
    seed_everything(seed, workers=True)

    # Transformer and Dataset
    transformer = StandardizeLabel()
    dataset = H5PYDatasetLabeled(dataset_path, transform=transformer)
    
    y = dataset.labels
    groups = dataset.sessions            
    
    # Main cross-validation loop
    for repeat in range(n_repeats):  # Repeating 10-fold cross-validation 5 times
        for i, (train_index, test_index) in enumerate(group_kfold.split(torch.arange(len(dataset)), y, groups)):
            training_set = torch.utils.data.Subset(dataset, train_index)
            test_set = torch.utils.data.Subset(dataset, test_index)
            
            training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
            
            model = LinearHeadBENDR(encoder_path, encoder_h=512, in_features=19, out_features=out_features)
            #model = FullBENDR(encoder_path, encoder_h=512, in_features=19, out_features=out_features)
            
            if freeze_encoder:
                for param in model.encoder.parameters():
                    param.requires_grad = False
                    
            model_checkpoint = ModelCheckpoint(monitor='val_loss', save_weights_only=True)
            logger = CSVLogger("/scratch/s194260/finetune_logs_all", name=name, prefix="Fold_{}_Repeat_{}".format(i, repeat))
            early_stopping = EarlyStopping(monitor='val_loss', patience=10)
            #progress_bar = RichProgressBar()
            
            trainer = Trainer(accelerator='auto', devices=n_device, max_epochs=num_epochs,
                               callbacks=[model_checkpoint, early_stopping], logger=logger,
                              check_val_every_n_epoch=1, log_every_n_steps=1)
            trainer.fit(model, train_dataloaders=training_loader, val_dataloaders=test_loader)
            trainer.test(model, dataloaders=test_loader, ckpt_path='best')
            
            # Get checkpoint path
            checkpoint_path = model_checkpoint.best_model_path
            os.remove(checkpoint_path)
            

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    CLI(finetune_cv)