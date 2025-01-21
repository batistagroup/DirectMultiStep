# Shape suffixes convention inspired by
# https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd

# B: batch size
# C: the length of the input on which conditioning is done
#    in our case input_max_length
# L: sequence length for decoder, in our case output_max_length
# M: memory length (length of sequence being attended to)
# D: model dimension (sometimes called d_model or embedding_dim)
# V: vocabulary size
# F: feed-forward subnetwork hidden size
# H: number of attention heads in a layer
# K: size of each attention key or value (sometimes called d_kv)


from typing import cast

import torch
import torch.nn as nn

Tensor = torch.Tensor


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        src_pad_idx: int,
        trg_pad_idx: int,
    ):
        super().__init__()

        self.decoder = decoder
        self.encoder = encoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src_BC: Tensor) -> Tensor:
        src_mask_B11C = (src_BC != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask_B11C

    def forward(self, src_BC: Tensor, trg_BL: Tensor, steps_B1: Tensor) -> Tensor:
        """
        src_BC is the product_item + one_sm_item combined
        trg_BL is the path_string of the corresponding route
        """
        src_mask_B11C = self.make_src_mask(src_BC.long())

        enc_src_BCD = self.encoder(src_BC.long(), src_mask_B11C, steps_B1)
        trg_mask = None  # this will trigger is_causal=True
        output_BLV = self.decoder(trg_BL, enc_src_BCD, src_mask_B11C, trg_mask_B1LL=trg_mask)
        return cast(Tensor, output_BLV)
