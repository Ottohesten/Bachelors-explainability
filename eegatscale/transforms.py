from random import randint
from typing import List
import numpy as np
import torch

class Standardize(object):
    def __init__(self, mean: float = 0.0, std: float = 1e-05, clip: float = 5e-05):
        self.mean = mean
        self.std = std
        self.clip = clip
        
    def __call__(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        return (torch.clamp(x, -self.clip, self.clip) - self.mean) / self.std
    
class StandardizeLabel(object):
    def __init__(self, mean: float = 0.0, std: float = 1e-05, clip: float = 5e-05):
        self.mean = mean
        self.std = std
        self.clip = clip
        
    def __call__(self, batch: List, training: bool = True) -> torch.Tensor:
        x, y = batch
        x = (torch.clamp(x, -self.clip, self.clip) - self.mean) / self.std
        return [x, y]
    
class Normalize(object):
    def __init__(self, clip: float = 5e-05):
        self.clip = clip
        
    def __call__(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        return (torch.clamp(x, -self.clip, self.clip) - x.min()) / (x.max() - x.min())*2 - 1

class NumpyToTensor(object):
    """Just a simple transform object for converting from numpy to tensor"""
    def __call__(self, x: np.ndarray, training: bool = True) -> torch.Tensor:
        return torch.from_numpy(x)


class ChannelExclude(object):
    def __init__(self, channels: None | List[int] = None, channel_dim: int = 1) -> None:
        self.channels = channels
        self.channel_dim = channel_dim

    def __call__(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        channel_size = x.shape[self.channel_dim]
        good_channels = torch.tensor([c for c in range(channel_size) if c not in self.channels])
        return torch.index_select(x, dim=self.channel_dim, index=good_channels)


class RandomTemporalCrop(object):
    """Randomly crop the temporal dimension.
    
    Args:
        max_crop_frac: the fraction of the temporal dimension to be cropped. On average
            approximately half will be removed from the start of the sequence and the
            other half from the end of the sequence.
        temporal_dim: index of the temporal dim to crop

    """
    def __init__(self, max_crop_frac: float = 0.25, temporal_dim: int = 1) -> None:
        if not (0 < max_crop_frac < 1):
            raise ValueError("Expected argument `max_crop_frac` to be between 0 and 1")
        self.max_crop_frac = max_crop_frac
        self.temporal_dim = temporal_dim

    def __call__(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        if training:  # only crop on training
            trial_len = x.shape[self.temporal_dim]
            crop_len = randint(int((1 - self.max_crop_frac) * trial_len), trial_len-1)
            offset = randint(0, trial_len - crop_len)

            # move to first axis and index and move back
            x = x.moveaxis(self.temporal_dim, 0)
            return x[offset:offset+crop_len].moveaxis(0, self.temporal_dim)
        return x

class ShuffleChannels(object):
    """Randomly shuffle the channels of the input tensor along the specified dimension.
    
    Args:
        channel_dim: index of the channel dim to shuffle

    """
    def __init__(self, channel_dim: int = 1) -> None:
        super().__init__()
        self.channel_dim = channel_dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.moveaxis(self.channel_dim, 0)
        return x[torch.randperm(x.shape[0])].moveaxis(0, self.channel_dim)