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

"""
Shape suffixes convention inspired by
https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd

B: batch size
C: the length of the input on which conditioning is done
   in our case input_max_length
L: sequence length for decoder, in our case output_max_length
M: memory length (length of sequence being attended to)
D: model dimension (sometimes called d_model or embedding_dim)
V: vocabulary size
F: feed-forward subnetwork hidden size
H: number of attention heads in a layer
K: size of each attention key or value (sometimes called d_kv)
"""

from typing import Any, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class ModelConfig:
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

        if hid_dim % n_heads != 0:
            raise ValueError(f"{hid_dim=} must be divisible by {n_heads=}")
        self.hid_dim = hid_dim  # dimensionality of embedding, D
        self.n_heads = n_heads  # number of heads
        self.n_layers = n_layers  # number of layers
        self.ff_mult = ff_mult  # multiplier for feedforward layer
        self.dropout = dropout
        self.attn_bias = attn_bias
        self.ff_activation: nn.Module
        if ff_activation == "gelu":
            self.ff_activation = nn.GELU()
        elif ff_activation == "relu":
            self.ff_activation = nn.ReLU()
        else:
            raise ValueError("attn_activation must be 'gelu' or 'relu'")

        self.ff_type: str
        self.top_k: int
        self.n_experts: int


class VanillaTransformerConfig(ModelConfig):
    ff_type = "vanilla"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()

        self.hid_dim = config.hid_dim
        self.n_heads = config.n_heads
        self.head_dim = config.hid_dim // config.n_heads

        self.query = nn.Linear(config.hid_dim, config.hid_dim, config.attn_bias)
        self.key = nn.Linear(config.hid_dim, config.hid_dim, config.attn_bias)
        self.value = nn.Linear(config.hid_dim, config.hid_dim, config.attn_bias)

        self.projection = nn.Linear(config.hid_dim, config.hid_dim)

        self.dropout = nn.Dropout(config.dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(
        self,
        query_BLD: Tensor,
        key_BMD: Tensor,
        value_BMD: Tensor,
        mask_B11M: Optional[Tensor] = None,
    ) -> Tensor:
        B, L, _ = query_BLD.shape
        Q_BLD = self.query(query_BLD)
        K_BMD = self.key(key_BMD)
        V_BMD = self.value(value_BMD)
        # Reshape into multiple heads
        Q_BHLD = Q_BLD.view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K_BHMD = K_BMD.view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V_BHMD = V_BMD.view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        if mask_B11M is not None:
            # Expand mask for all heads
            mask_BHLM = mask_B11M.expand(B, self.n_heads, L, -1)
            is_causal = False
        else:
            mask_BHLM = None
            is_causal = True

        attn_output_BHLD = nn.functional.scaled_dot_product_attention(
            query=Q_BHLD,
            key=K_BHMD,
            value=V_BHMD,
            attn_mask=mask_BHLM,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal,
            # scale=self.scale.item(),
        )
        attn_output_BLD = (
            attn_output_BHLD.permute(0, 2, 1, 3).contiguous().view(B, L, self.hid_dim)
        )
        output_BLD = cast(Tensor, self.projection(attn_output_BLD))
        return output_BLD


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.fc_1 = nn.Linear(config.hid_dim, config.ff_mult * config.hid_dim)
        self.activ = config.ff_activation
        self.fc_2 = nn.Linear(config.hid_dim * config.ff_mult, config.hid_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x_BLD: Tensor) -> Tensor:
        x_BLF = self.dropout(self.activ(self.fc_1(x_BLD)))
        x_BLD = self.fc_2(x_BLF)
        return x_BLD


class NoisyTopkRouter(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.top_k = config.top_k
        self.topkroute_linear = nn.Linear(config.hid_dim, config.n_experts)
        self.noise_linear = nn.Linear(config.hid_dim, config.n_experts)

    def forward(self, x_BLD: Tensor) -> Tuple[Tensor, Tensor]:
        """
        E - number of experts
        K - top_k (it's different from K in MHA!)
        """
        logits_BLE = self.topkroute_linear(x_BLD)
        noise_logits_BLE = self.noise_linear(x_BLD)
        # Adding scaled unit gaussian noise to the logits
        noise_BLE = torch.randn_like(logits_BLE) * F.softplus(noise_logits_BLE)
        noisy_logits_BLE = logits_BLE + noise_BLE

        top_k_logits_BLE, indices_BLK = noisy_logits_BLE.topk(self.top_k, dim=-1)
        zeros_BLE = torch.full_like(noisy_logits_BLE, float("-inf"))
        # creating a sparse tensor with top-k logits
        sparse_logits_BLE = zeros_BLE.scatter(-1, indices_BLK, top_k_logits_BLE)
        router_output_BLE = F.softmax(sparse_logits_BLE, dim=-1)
        return router_output_BLE, indices_BLK


class Expert(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hid_dim, config.ff_mult * config.hid_dim),
            config.ff_activation,
            nn.Linear(config.ff_mult * config.hid_dim, config.hid_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x_BLD: Tensor) -> Tensor:
        return self.net(x_BLD)  # type: ignore


class SparseMoE(nn.Module):
    def __init__(self, config: ModelConfig):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])
        self.top_k = config.top_k

    def forward(self, x_BLD: Tensor) -> Tensor:
        """
        E - number of experts
        K - top_k (it's different from K in MHA!)
        S - how many times the expert is selected
        """
        gating_output_BLE, indices_BLK = self.router(x_BLD)
        final_output_BLD = torch.zeros_like(x_BLD)

        flat_x_FD = x_BLD.view(-1, x_BLD.size(-1))  # [B*L, D], define B*L=F
        flat_gating_output_FE = gating_output_BLE.view(-1, gating_output_BLE.size(-1))

        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask_BL = (indices_BLK == i).any(dim=-1)
            flat_mask_F = expert_mask_BL.view(-1)

            if flat_mask_F.any():
                expert_input_SD = flat_x_FD[flat_mask_F]  # S = sum(flat_mask_F)
                expert_output_SD = expert(expert_input_SD)

                # Extract and apply gating scores, [S] -> [S, 1]
                gating_scores_S1 = flat_gating_output_FE[flat_mask_F, i].unsqueeze(1)
                weighted_output_SD = expert_output_SD * gating_scores_S1
                # Update final output additively by indexing and adding
                final_output_BLD[expert_mask_BL] += weighted_output_SD
                # final_output_BLD[expert_mask_BL] has shape SD!

        return final_output_BLD


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()

        self.attn_ln = nn.LayerNorm(config.hid_dim)
        self.ff_ln = nn.LayerNorm(config.hid_dim)
        self.attention = MultiHeadAttentionLayer(config, device)
        self.mlp: nn.Module
        if config.ff_type == "moe":
            self.mlp = SparseMoE(config)
        elif config.ff_type == "vanilla":
            self.mlp = PositionwiseFeedforwardLayer(config)
        else:
            raise ValueError(f"Unknown feedforward type: {config.ff_type}")
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_BCD: Tensor, src_mask_B11C: Tensor) -> Tensor:
        attn_output_BCD = self.attention(input_BCD, input_BCD, input_BCD, src_mask_B11C)
        src_BCD = self.attn_ln(input_BCD + self.dropout(attn_output_BCD))
        ff_out_BCD = self.mlp(src_BCD)
        final_out_BLD = self.ff_ln(src_BCD + self.dropout(ff_out_BCD))
        return cast(Tensor, final_out_BLD)


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()

        self.hid_dim = config.hid_dim  # D
        self.input_dim = config.input_dim  # D
        self.tok_embedding = nn.Embedding(self.input_dim, config.hid_dim)  # V, D
        self.pos_embedding = nn.Embedding(
            config.input_max_length, config.hid_dim
        )  # C, D
        self.step_embedding = nn.Embedding(1, config.hid_dim)  # 1, D

        self.layers = nn.ModuleList(
            [EncoderLayer(config=config, device=device) for _ in range(config.n_layers)]
        )

        self.dropout = nn.Dropout(config.dropout)
        self.max_length = config.input_max_length
        self.scale = torch.sqrt(torch.FloatTensor([config.hid_dim])).to(device)

    def forward(
        self, src_BC: Tensor, src_mask_B11C: Tensor, steps_B1: Tensor
    ) -> Tensor:
        B, C = src_BC.shape
        tok_emb_BCD = self.tok_embedding(src_BC) * self.scale.to(src_BC)
        # below [C] -> [1, C] -> [B, C]
        pos_BC = torch.arange(0, C).unsqueeze(0).repeat(B, 1).to(src_BC)
        pos_emb_BCD = self.pos_embedding(pos_BC)
        # [C] -> [1, C] -> [B, C]
        step_BC = torch.zeros(C).unsqueeze(0).repeat(B, 1).long().to(src_BC)
        step_emb_BCD = self.step_embedding(step_BC) * steps_B1.view(-1, 1, 1)
        src_BCD = self.dropout(tok_emb_BCD + pos_emb_BCD + step_emb_BCD)
        for layer in self.layers:
            src_BCD = layer(src_BCD, src_mask_B11C)
        return cast(Tensor, src_BCD)


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()
        self.self_attn_ln = nn.LayerNorm(config.hid_dim)
        self.enc_attn_ln = nn.LayerNorm(config.hid_dim)
        self.ff_ln = nn.LayerNorm(config.hid_dim)
        self.self_attn = MultiHeadAttentionLayer(config, device)
        self.encoder_attn = MultiHeadAttentionLayer(config, device)
        self.mlp: nn.Module
        if config.ff_type == "moe":
            self.mlp = SparseMoE(config)
        elif config.ff_type == "vanilla":
            self.mlp = PositionwiseFeedforwardLayer(config)
        else:
            raise ValueError(f"Unknown feedforward type: {config.ff_type}")
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        trg_BLD: Tensor,
        enc_src_BCD: Tensor,
        src_mask_B11C: Tensor,
        trg_mask_B1LL: Tensor,
    ) -> Tensor:
        self_attn_BLD = self.self_attn(trg_BLD, trg_BLD, trg_BLD, trg_mask_B1LL)
        trg_BLD = self.self_attn_ln(trg_BLD + self.dropout(self_attn_BLD))
        # Encoder-Decoder Attetion
        enc_attn_BLD = self.encoder_attn(
            trg_BLD, enc_src_BCD, enc_src_BCD, src_mask_B11C
        )
        trg_BLD = self.enc_attn_ln(trg_BLD + self.dropout(enc_attn_BLD))
        ff_out_BLD = self.mlp(trg_BLD)
        trg_BLD = self.ff_ln(trg_BLD + self.dropout(ff_out_BLD))
        return trg_BLD


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()

        self.hid_dim = config.hid_dim
        self.max_length = config.output_max_length
        self.tok_embedding = nn.Embedding(config.output_dim, config.hid_dim)
        self.pos_embedding = nn.Embedding(config.output_max_length, config.hid_dim)

        self.layers = nn.ModuleList(
            [DecoderLayer(config=config, device=device) for _ in range(config.n_layers)]
        )

        self.fc_out = nn.Linear(config.hid_dim, config.output_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.scale = torch.sqrt(torch.FloatTensor([config.hid_dim])).to(device)

    def forward(
        self,
        trg_BL: Tensor,
        enc_src_BCD: Tensor,
        src_mask_B11C: Tensor,
        trg_mask_B1LL: Optional[Tensor] = None,
    ) -> Tensor:
        B, L = trg_BL.shape
        # below: [L] -> [1, L] -> [B, L]
        pos_BL = torch.arange(0, L).unsqueeze(0).repeat(B, 1).to(trg_BL)
        tok_emb_BLD = self.tok_embedding(trg_BL) * self.scale.to(trg_BL)
        pos_emb_BLD = self.pos_embedding(pos_BL)
        trg_BLD = self.dropout(tok_emb_BLD + pos_emb_BLD)
        for layer in self.layers:
            trg_BLD = layer(trg_BLD, enc_src_BCD, src_mask_B11C, trg_mask_B1LL)
        output_BLV = self.fc_out(trg_BLD)
        return cast(Tensor, output_BLV)


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
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
        output_BLV = self.decoder(
            trg_BL, enc_src_BCD, src_mask_B11C, trg_mask_B1LL=trg_mask
        )
        return cast(Tensor, output_BLV)
