"""
execute using command:
    python scripts/pretrain.py -c configs/pretrain.yaml fit

"""
from typing import Any, Callable, Sequence, List

import h5py
import torch
from pytorch_lightning import LightningDataModule, cli
from pytorch_lightning.cli import SaveConfigCallback
from torch.utils.data import DataLoader, Dataset
import os

from eegatscale.utils import flatten_namespace

class H5PYDataset(Dataset):
    def __init__(self, path: str):
        if os.path.isdir(path):
            self.paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.hdf5')]
        else:
            raise ValueError("Path should be a directory")
        
        self.paths.sort()
        self.key = 'data'
        self.lengths = [self._get_file_length(path) for path in self.paths]
        self.cumulative_lengths = self._compute_cumulative_lengths(self.lengths)

    def _get_file_length(self, path):
        with h5py.File(path, 'r') as file:
            return file[self.key].shape[0]

    def _compute_cumulative_lengths(self, lengths):
        cumulative_lengths = [0]
        for length in lengths:
            cumulative_lengths.append(cumulative_lengths[-1] + length)
        return cumulative_lengths

    def __len__(self):
        return self.cumulative_lengths[-1]

    def _load_data(self, path, local_index):
        with h5py.File(path, 'r') as file:
            return file[self.key][local_index]

    def __getitem__(self, global_index: int):
        # If global_index is out of bounds, raise an error
        if global_index < 0 or global_index >= len(self):
            raise IndexError(f"Index {global_index} out of bounds for dataset of length {len(self)}")
        
        file_index = self._find_file_index(global_index)
        local_index = global_index - self.cumulative_lengths[file_index]
        data = self._load_data(self.paths[file_index], local_index)
        return data

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

class WrapperDataModule(LightningDataModule):
    def __init__(
        self, 
        trainset_path: str, 
        valset_path: None | str = None,
        batch_size: int = 2, 
        num_workers: int = 1, 
        batch_transforms: None | Sequence[Callable] = None
    ) -> None:
        super().__init__()
    
        self.training_dataset = H5PYDataset(trainset_path)
        self.validation_dataset = H5PYDataset(valset_path) if valset_path is not None else None

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch_transforms = batch_transforms

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.training_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False
        )

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.batch_transforms is not None:
            for t in self.batch_transforms:
                batch = t(batch, training=self.trainer.training)
        return batch
    
    def teardown(self, stage: str) -> None:
        self.training_dataset.teardown()
        if self.validation_dataset is not None:
            self.validation_dataset.teardown()

class WandbSaveConfigCallback(SaveConfigCallback):
    """Make sure that config gets correctly saved to wandb"""
    def save_config(self, trainer, pl_module, stage: str) -> None:
        trainer.logger.log_hyperparams(flatten_namespace(self.config))

class BendrCLI(cli.LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "model.init_args.encoder.init_args.encoder_h",
            "model.init_args.contextualizer.init_args.in_features"
        )

    @staticmethod
    def configure_optimizers(lightning_module, optimizer, lr_scheduler) -> Any:
        if lr_scheduler is None:
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer=optimizer,
                    max_lr=optimizer.param_groups[-1]["lr"],
                    total_steps=lightning_module.trainer.estimated_stepping_batches
                ),
                "interval": "step"
            }
        return [optimizer], [lr_scheduler]


if __name__ == "__main__":
    import mne
    mne.set_log_level("CRITICAL")
    BendrCLI(
        datamodule_class=WrapperDataModule, 
        save_config_callback=WandbSaveConfigCallback,
        save_config_kwargs={"overwrite": True}
    )
