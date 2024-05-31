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
from DirectMultiStep.Models.Architecture import Encoder, Decoder, Seq2Seq, ModelConfig


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def determine_device(allow_mps: bool = False) -> str:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if allow_mps and torch.backends.mps.is_available()
        else "cpu"
    )
    return device


def prepare_model(enc_config: ModelConfig, dec_config: ModelConfig) -> nn.Module:
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
