# MIT License

# Copyright (c) 2024 Batista Lab (Yale University)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
from .Architecture import Encoder, Decoder, Seq2Seq


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def determine_device(allow_mps: bool = False) -> str:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if allow_mps and torch.backends.mps.is_available() else "cpu"
    )
    return device


class VanillaTransformerConfig:
    ff_type = "vanilla"

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        input_max_length: int,
        output_max_length: int,
        pad_index: int,
        hid_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        attn_bias: bool = False,
        ff_activation: str = "gelu",
    ):
        self.input_dim = input_dim  # Total input dimensions including padding
        self.input_max_length = input_max_length
        self.pad_index = pad_index

        self.output_dim = output_dim
        self.output_max_length = output_max_length

        assert hid_dim % n_heads == 0, "hid_dim must be divisible by n_heads"
        self.hid_dim = hid_dim  # dimensionality of embedding, D
        self.n_heads = n_heads  # number of heads
        self.n_layers = n_layers  # number of layers
        self.ff_mult = ff_mult  # multiplier for feedforward layer
        self.dropout = dropout
        self.attn_bias = attn_bias
        if ff_activation == "gelu":
            self.ff_activation = nn.GELU()
        elif ff_activation == "relu":
            self.ff_activation = nn.ReLU()
        else:
            raise ValueError("attn_activation must be 'gelu' or 'relu'")



def prepare_model(enc_config, dec_config):
    device = torch.device(determine_device())
    encoder = Encoder(config=enc_config, device=device)
    decoder = Decoder(config=dec_config, device=device)
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=enc_config.pad_index,
        trg_pad_idx=dec_config.pad_index,
    )
    model.to(device)
    torch.compile(model)
    print(f"The model has {count_parameters(model):,} trainable parameters")
    return model


if __name__ == "__main__":
    pass
