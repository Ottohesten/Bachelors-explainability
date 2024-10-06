import torch
from mne.io import read_raw_edf, read_raw_fif
from torch import nn
from transformers import GPT2Config, GPT2Model

####
# Implementation of NeuroGPT, a model that combines EEG Conformer and GPT-2. https://arxiv.org/abs/2311.03764
# Still a work-in-progress, currently does not work as intended. 
# The problem is likely to be due to the way the loss is calculated with the learnable token.
####


class NeuroGPT(nn.Module):
    def __init__(
        self,
        num_channels=30,
        signal_length=512,
        embed_dim=1080,
        num_heads=10,
        num_layers=6,
        temporal_kernel=(1, 25),
        gpt_n_layer=6,
        gpt_n_head=16,
        gpt_n_embd=1024,
    ):
        super().__init__()

        self.temporal_encoder = nn.Conv2d(
            in_channels=1, out_channels=embed_dim, kernel_size=temporal_kernel
        )

        self.spatial_encoder = nn.Conv2d(
            in_channels=embed_dim, out_channels=embed_dim, kernel_size=(num_channels, 1)
        )

        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
                for _ in range(num_layers)
            ]
        )

        self.learnable_token = nn.Parameter(torch.empty(size=(num_channels, signal_length)), requires_grad=True)

        # same avg pooling as EEG Conformer
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))

        self.linear_proj = nn.Linear(embed_dim, gpt_n_embd)

        gpt_configuration = GPT2Config(
            n_layer=gpt_n_layer, n_head=gpt_n_head, n_embd=gpt_n_embd
        )
        gpt = GPT2Model(gpt_configuration)
        self.gpt_h = gpt.h
        self.gpt_ln_f = gpt.ln_f

        self.criterion = nn.L1Loss()

    def convolutional_encoder(self, chunk):
        x = chunk.unsqueeze(0)
        x = self.temporal_encoder(x)  # [embed_dim, n_ch, sig_len]
        x = self.spatial_encoder(x)   # [embed_dim, 1, sig_len]
        x = self.avg_pool(x)          # [embed_dim, 1, sig_len]
        x = x.squeeze()               # [embed_dim, sig_len]
        x = nn.Dropout(p=0.5)(x)      # [embed_dim, sig_len]
        x = x.permute(1, 0)           # [embed_dim, sig_len]

        for attention_layer in self.attention_layers:
            x, _ = attention_layer(x, x, x)

        return x


    def forward(self, x):

        h = torch.stack([self.convolutional_encoder(chunk) for chunk in x])        
        h = self.linear_proj(h)

        M = self.linear_proj(self.convolutional_encoder(self.learnable_token))

        loss = 0.0
        for i in range(1, len(h)-1):
            h_dupe = h.clone()
            h_dupe[i:] = 0
            h_dupe[i] = M

            attention_mask = torch.zeros(len(h))
            attention_mask[:i+1] = 1

            h_dupe = h_dupe.permute(1,0,2)
            for block in self.gpt_h:
                h_dupe = block(h_dupe, attention_mask=attention_mask)[0]

            h_dupe = h_dupe.permute(1,0,2)
            h_dupe = self.gpt_ln_f(h_dupe)
            print("Norm of M", torch.norm(M))

            loss += self.criterion(h_dupe[i], h[i])
        return loss

def divide_signal(input_tensor, chunk_length, overlap=0.1):
    """
    Divide a PyTorch tensor of shape [C, S] into N chunks of length T with overlap.

    Parameters:
        input_tensor (torch.Tensor): Input tensor of shape [C, S].
        chunk_length (int): Length of each chunk (T).
        overlap (float): Overlap ratio between chunks (default is 0.1).

    Returns:
        torch.Tensor: A tensor containing the divided chunks along a new dimension.
    """
    assert 0 <= overlap < 1, "Overlap must be between 0 and 1."

    _, signal_length = input_tensor.shape
    overlap_length = int(chunk_length * overlap)
    step_size = chunk_length - overlap_length

    # Calculate the number of chunks
    num_chunks = (signal_length - chunk_length) // step_size + 1

    # Create empty tensor to store the chunks
    chunks = torch.zeros(num_chunks, input_tensor.shape[0], chunk_length)

    # Populate the chunks tensor
    for i in range(num_chunks):
        start = i * step_size
        end = start + chunk_length
        chunks[i, :, :] = input_tensor[:, start:end]

    return chunks


def batch_sequence(file_path, chunk_len_s=2, overlap=0.1, n_chunks=32):
    if file_path.endswith(".edf"):
        raw = read_raw_edf(file_path)
    elif file_path.endswith(".fif"):
        raw = read_raw_fif(file_path)
    else:
        raise ValueError("File format not supported")

    signal = torch.tensor(raw.get_data())

    sampling_freq = int(raw.info["sfreq"])
    chunk_length = chunk_len_s * sampling_freq
    num_channels = signal.shape[0]

    chunk_seq = divide_signal(
        input_tensor=signal, chunk_length=chunk_length, overlap=overlap
    )

    padding = torch.zeros(
        size=(
            (n_chunks - (len(chunk_seq) % n_chunks)) % n_chunks,
            num_channels,
            chunk_length,
        )
    )
    chunk_seq = torch.cat((chunk_seq, padding), dim=0)

    return chunk_seq.view(-1, n_chunks, num_channels, chunk_length)
