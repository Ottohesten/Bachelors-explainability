from copy import deepcopy
from math import ceil
from typing import Any, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torch.nn import functional as F

from eegatscale.layers import Permute
from eegatscale.utils import _make_mask, _make_span_from_seeds


class BendrEncoder(nn.Module):
    """
    BENDR convolutional encoder module.

    Args:
        in_features: number of input features / channels
        encoder_h: number of output features in each convolutional operator e.g. number of output features
        enc_width: list of integers indicating the kernel widths for the encoder
        dropout: probability for dropout layer
        projection_head: if projection head should be added such that the number output features should be projected
            to be the same as the number of input features
        enc_downsample: list of integers indicating the strides for the encoder
        grad_frac: float to multiply onto all gradients

    Example:
        >>> from eegatscale.bendr import BendrEncoder
        >>> import torch
        >>> encoder = BendrEncoder(in_features = 30)
        >>> signal = torch.randn(10, 30, 100)  # batch_size x in_features x signal length
        >>> out = encoder(signal)
        >>> print(out.shape)
        torch.Size([10, 256, 2])  # batch_size x encoder_h x out length
    """
    def __init__(
        self,
        in_features,
        encoder_h: int = 256,
        enc_width: Tuple[int, ...] = (3, 2, 2, 2, 2, 2),
        dropout: float = 0.0,
        projection_head: bool = False,
        enc_downsample: Tuple[int, ...] = (3, 2, 2, 2, 2, 2),
        grad_frac: float = 1.0
    ) -> None:
        super().__init__()
        if not isinstance(enc_width, tuple):
            raise ValueError("Expected argument `enc_width` to be of type tuple")
        if not isinstance(enc_downsample, tuple):
            raise ValueError("Expected argument `enc_downsample` to be of type tuple")
        if len(enc_width) != len(enc_downsample):
            raise ValueError("Expected argum,ent `enc_width` and `enc_downsample` to have same length")
        self.in_features = in_features
        self.encoder_h = encoder_h

        # center convolutions
        enc_width = [e if e % 2 else e+1 for e in enc_width]        
        self.encoder = nn.Sequential()
        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample, strict=True)):
            self.encoder.add_module(
                f"Encoder_{i}",
                nn.Sequential(
                    nn.Conv1d(in_features, encoder_h, width, stride=downsample, padding=width // 2),
                    nn.Dropout1d(dropout),  # changed from 2d to 1d compared to bendr
                    nn.GroupNorm(encoder_h // 2, encoder_h),
                    nn.GELU(),
                )
            )
            in_features = encoder_h
        self.enc_width = enc_width
        self.enc_downsample = enc_downsample

        if projection_head:
            self.encoder.add_module(
                "projection-1",
                nn.Sequential(
                    nn.Conv1d(in_features, in_features, 1),
                    nn.Dropout1d(dropout*2),  # changed from 2d to 1d compared to bendr
                    nn.GroupNorm(in_features // 2, in_features),
                    nn.GELU()
                )
            )
        self.enc_downsample = enc_downsample
        self.out_features = in_features if projection_head else encoder_h

        if grad_frac < 1.0:
            self.register_backward_hook(
                lambda module, in_grad, out_grad: tuple(grad_frac * ig for ig in in_grad)
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def downsampling_factor(self, input_length: int) -> int:
        """ Calculates the size of the output lenght for a given input length."""
        for factor in self.enc_downsample:
            input_length = ceil(input_length / factor)
        return input_length


class BendrContextualizer(nn.Module):
    """Bendr contextualizer.

    This class is in charge of ...

    Args:
        in_features:
        dim_feedforward:
        nheads:
        layers:
        dropout:
        activation:
        position_encoder:
        layer_drop:
        mask_p_t: probability that a token is masked along the temporal dimension
        mask_p_c: probability that a token is masked along the channel dimension
        mask_t_span: the length of each mask along the temporal dimension, meaning that if index `i` is sampled to
            be masked then all elements until `i+mask_t_span` will also be masked.
        mask_c_span: the length of each mask along the temporal dimension, meaning that if index `i` is sampled to
            be masked then all elements until `i+mask_c_span` will also be masked.
        start_token:
        finetuning:

    """

    def __init__(
        self,
        in_features: int,
        dim_feedforward: int = 3076,
        nheads: int = 8,
        layers: int = 8,
        dropout: float = 0.15,
        activation: str = 'gelu',
        position_encoder: int = 25,
        layer_drop: float = 0.0,
        mask_p_t: float = 0.1,
        mask_p_c: float = 0.004,
        mask_t_span: int = 6,
        mask_c_span: int = 64,
        start_token: int = -5,
        finetuning: bool = False
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.transformer_dim = in_features * 3
        
        self.finetuning = finetuning

        # default transformers need input of shape (seq, batch, feature)
        encoder = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )

        # remove self-attention norms
        encoder.norm1 = nn.Identity()
        encoder.norm2 = nn.Identity()
        self.norm = nn.LayerNorm(self.transformer_dim)

        self.transformer_layers = nn.ModuleList([deepcopy(encoder) for _ in range(layers)])
        self.layer_drop = layer_drop
        self.mask_p_t = mask_p_t
        self.mask_p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        self.start_token = start_token

        # Initialize replacement vector with 0's
        self.mask_replacement = nn.Parameter(
            torch.normal(0, in_features ** (-0.5), size=(in_features,), requires_grad=True)
        )

        self.position_encoder = position_encoder > 0
        if position_encoder:
            conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
            nn.init.normal_(conv.weight, mean=0, std=2 / self.transformer_dim)
            nn.init.constant_(conv.bias, 0)
            conv = nn.utils.parametrizations.weight_norm(conv, dim=2)
            self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            Permute(0, 2, 1),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute(0, 2, 1),
            nn.Conv1d(in_features, self.transformer_dim, 1),
            Permute(2, 0, 1),
        )

        self.output_layer = nn.Conv1d(self.transformer_dim, in_features, 1)
        self.apply(self.init_best_params)

    def init_best_params(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data = 0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data

    def forward(self, x: Tensor, mask_t: Tensor | None = None, mask_c: Tensor | None = None) -> Tensor:
        batch_size, num_features, sequence_length = x.shape
        if self.training and self.finetuning:
            if mask_t is None and self.mask_p_t > 0:
                mask_t = _make_mask(
                    (batch_size, sequence_length), prob=self.mask_p_t, total=sequence_length, span=self.mask_t_span
                )  # batch_size x sequence_length
            if mask_c is None and self.mask_p_c > 0:
                mask_c = _make_mask(
                    (batch_size, num_features), prob=self.mask_p_c, total=num_features, span=self.mask_c_span
                )  # batch_size x num_features

        if mask_t is not None:
            x.transpose(2,1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0.0

        if self.position_encoder:
            x = x + self.relative_position(x)  # batch_size x num_features x sequence_length
        x = self.input_conditioning(x)  # sequence_length x batch_size x transformer_dim

        if self.start_token is not None:
            in_token = self.start_token * torch.ones(1, x.shape[1], x.shape[2], device=x.device)
            x = torch.cat([in_token, x], dim=0)  # (sequence_length + 1) x batch_size x transformer_dim

        for layer in self.transformer_layers:
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x)  # (sequence_length + 1) x batch_size x transformer_dim

        x = x.permute(1, 2, 0)  # batch_size x transformer_dim x (sequence_length + 1)
        return self.output_layer(x)  # batch_size x num_features x (sequence_length + 1)


class Bendr(LightningModule):
    """
    Main BENDR model class.

    Args:
        encoder: Instance of a encoder module
        contextualizer: Instance of a contextualizer module
        mask_rate: probability of elements being masked
        mask_span: the length of masked spans
        temp: temperature used to calibrate cosine similarity
        activation_weight: weight term for the mean activation that is added to the overall loss
        num_negatives: number of negatives sampled

    """
    def __init__(
        self,
        encoder: nn.Module,
        contextualizer: nn.Module,
        mask_rate: float = 0.1,
        mask_span: int = 6,
        temp: float = 0.5,
        activation_weight: float = 0.0001,
        num_negatives: int = 100,
        checkpoint: str = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.contextualizer = contextualizer
        self.mask_rate = mask_rate
        self.mask_span = mask_span
        self.temp = temp
        self.activation_weight = activation_weight
        self.num_negatives = num_negatives
        self.loss_fn = nn.CrossEntropyLoss()
        
        if checkpoint:
            state_dict = torch.load(checkpoint)['state_dict']
            encoder_state = {k[8:]: v for k, v in state_dict.items() if k.startswith("encoder")}
            contextualizer_state = {k[15:]: v for k, v in state_dict.items() if k.startswith("contextualizer")}
            self.encoder.load_state_dict(encoder_state)
            self.contextualizer.load_state_dict(contextualizer_state)

    def forward(self, x: Tensor) -> Tensor:  # batch_size x in_features x signal length
        # encode each window
        z = self.encoder(x)  # batch_size x encoder_h x out length
        unmasked_z = z.clone()

        batch_size, _, num_samples = z.shape
        if self.training:
            mask = _make_mask((batch_size, num_samples), prob=self.mask_rate, total=num_samples, span=self.mask_span)
        else:
            mask = torch.zeros(batch_size, num_samples, requires_grad=False, dtype=torch.bool)
            half_avg_num_seeds = max(1, int(num_samples * self.mask_rate * 0.5))
            spans = _make_span_from_seeds(
                seeds=(num_samples // half_avg_num_seeds) * np.arange(half_avg_num_seeds).astype(int), 
                span=self.mask_span,
                total=num_samples,
            )
            mask[:, spans] = True

        # get context meaning pass through transformer
        context = self.contextualizer(z, mask_t=mask) # batch_size x num_features x (sequence_length + 1)

        negatives, _ = self.generate_negatives(z)  # batch_size x (sequence_length + 1) x num_negatives x num_features

        logits = self.calculate_similarity(unmasked_z, context, negatives)
        return logits, z, mask

    def generate_negatives(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, num_features, sequence_length = z.shape
        z_k = z.permute(0, 2, 1).reshape(-1, num_features)

        with torch.no_grad():
            negative_idx = torch.randint(0, sequence_length-1, size=(batch_size, sequence_length * self.num_negatives))
            for i in range(1, batch_size):
                negative_idx[i] += i * sequence_length

        z_k = z_k[negative_idx.view(-1)].view(batch_size, sequence_length, self.num_negatives, num_features)
        return z_k, negative_idx

    def calculate_similarity(self, z: Tensor, context: Tensor, negatives: Tensor) -> Tensor:
        context = context[..., 1:].permute(0, 2, 1).unsqueeze(-2)  # bs x sequence_lenght x 1 num_features
        z = z.permute(0, 2, 1).unsqueeze(-2)  # bs x sequence_length x 1 x num_features

        # prevent division by zero if everything matches
        negative_in_target = (context == negatives).all(-1)  # bs x sequence_length x num_negatives
        targets = torch.cat([context, negatives], dim=-2)  # bs x sequence_length x (num_negatives + 1) x num_features

        logits = F.cosine_similarity(z, targets, dim=-1) / self.temp  # bs x sequence_length x (num_negatives + 1)
        if negative_in_target.any():
            logits[1:][negative_in_target] = float("-inf")
        return logits.view(-1, logits.shape[-1])  # (bs * sequence_length) x (num_negatives + 1)

    def calculate_loss_and_metrics(self, x: Tensor, logits: Tensor, z: Tensor):
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        # Note the loss_fn here integrates the softmax as per the normal classification pipeline (leveraging logsumexp)
        loss = self.loss_fn(logits, labels) + self.activation_weight * z.pow(2).mean()
        acc = (logits.argmax(dim=1) == labels).float().mean()
        return loss, {"acc": acc}

    def _step(self, batch):
        logits, z, mask = self(batch)
        loss, metrics = self.calculate_loss_and_metrics(batch, logits, z)
        return loss, metrics

    def training_step(self, batch, batch_idx) -> Tensor:
        loss, metrics = self._step(batch)
        self.log("loss", loss)
        if len(metrics):
            self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        loss, metrics = self._step(batch)
        self.log("val_loss", loss)
        if len(metrics):
            self.log_dict({f"val_{k}": v for k,v in metrics.items()})

