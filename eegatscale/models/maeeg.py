from eegatscale.models.bendr import Bendr
from torch import nn, Tensor
import torch
from eegatscale.utils import _make_mask, _make_span_from_seeds
import numpy as np

class MAEEG(Bendr):
    def __init__(
        self,
        encoder: nn.Module,
        contextualizer: nn.Module,
        mask_rate: float = 0.1,
        mask_span: int = 6,
        temp: float = 0.5,
        activation_weight: float = 0.0001,
        num_negatives: int = 100,
        # new arguments compared to bendr base model
        sequence_length: int = 15360,
        spatial_kernel_size: int = 5,
    ) -> None:
        super().__init__(
            encoder, contextualizer, mask_rate, mask_span, temp, activation_weight, num_negatives
        )
        out_length = self.encoder.downsampling_factor(sequence_length)
        self.temporal_recon = nn.Linear(out_length, sequence_length)
        self.spatial_recon = nn.Conv1d(
            self.contextualizer.in_features, 
            self.encoder.in_features, 
            spatial_kernel_size,
            padding=spatial_kernel_size // 2,
            stride=1,
        )
        self.loss_fn = nn.CosineSimilarity()

    def forward(self, x: Tensor) -> Tensor:  # batch_size x in_features x signal length
        # encode each window
        z = self.encoder(x)  # batch_size x encoder_h x out length

        batch_size, _, num_samples = z.shape
        if self.training:
            mask = _make_mask((batch_size, num_samples), prob=self.mask_rate, total=num_samples, span=self.mask_span)
        else:
            mask = torch.zeros(batch_size, num_samples, requires_grad=False, dtype=torch.bool)
            half_avg_num_seeds = max(1, int(num_samples * self.mask_rate * 0.5))
            spans = _make_span_from_seeds(
                (num_samples // half_avg_num_seeds) * np.arange(half_avg_num_seeds).astype(int), self.mask_span
            )
            mask[:, spans] = True

        # get context meaning pass through transformer
        context = self.contextualizer(z, mask_t=mask) # batch_size x num_features x (sequence_length + 1)

        # remove start token
        context = context[:, :, 1:]  # batch_size x num_features x sequence_length

        recon = self.temporal_recon(context)
        recon = self.spatial_recon(recon)

        return recon, z, mask
    
    def calculate_loss_and_metrics(self, x: Tensor, logits: Tensor, z: Tensor):
        loss = (1 - self.loss_fn(x, logits)).mean() + self.activation_weight * z.pow(2).mean()
        metrics = { }
        return loss, metrics


