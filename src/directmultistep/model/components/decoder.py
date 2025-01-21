from typing import cast

import torch
import torch.nn as nn

from directmultistep.model.components.attention import MultiHeadAttentionLayer
from directmultistep.model.components.moe import PositionwiseFeedforwardLayer, SparseMoE

Tensor = torch.Tensor
activation_dict = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
}


class DecoderLayer(nn.Module):
    """A single layer of the decoder.

    Shape suffixes convention:
        B: batch size
        C: the length of the input on which conditioning is done
           (in our case input_max_length)
        L: sequence length for decoder, in our case output_max_length
        D: model dimension (sometimes called d_model or embedding_dim)
    """

    def __init__(
        self,
        hid_dim: int,
        n_heads: int,
        dropout: float,
        attn_bias: bool,
        ff_mult: int,
        ff_activation: str,
    ) -> None:
        """Initializes the DecoderLayer.

        Args:
            hid_dim: The hidden dimension size.
            n_heads: The number of attention heads.
            dropout: The dropout rate.
            attn_bias: Whether to use bias in the attention layers.
            ff_mult: The feed-forward expansion factor.
            ff_activation: The activation function type.
        """
        super().__init__()
        self.self_attn_ln = nn.LayerNorm(hid_dim)
        self.enc_attn_ln = nn.LayerNorm(hid_dim)
        self.ff_ln = nn.LayerNorm(hid_dim)
        self.self_attn = MultiHeadAttentionLayer(
            hid_dim=hid_dim,
            n_heads=n_heads,
            dropout=dropout,
            attn_bias=attn_bias,
        )
        self.encoder_attn = MultiHeadAttentionLayer(
            hid_dim=hid_dim,
            n_heads=n_heads,
            dropout=dropout,
            attn_bias=attn_bias,
        )
        self.mlp: nn.Module = PositionwiseFeedforwardLayer(
            hid_dim=hid_dim,
            ff_mult=ff_mult,
            ff_activation=activation_dict[ff_activation],
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        trg_BLD: Tensor,
        enc_src_BCD: Tensor,
        src_mask_B11C: Tensor,
        trg_mask_B1LL: Tensor,
    ) -> Tensor:
        """Forward pass of the DecoderLayer.

        Args:
            trg_BLD: The target sequence tensor of shape (B, L, D).
            enc_src_BCD: The encoder output tensor of shape (B, C, D).
            src_mask_B11C: The source mask tensor of shape (B, 1, 1, C).
            trg_mask_B1LL: The target mask tensor of shape (B, 1, L, L).

        Returns:
            The output tensor of shape (B, L, D).
        """
        self_attn_BLD = self.self_attn(trg_BLD, trg_BLD, trg_BLD, trg_mask_B1LL)
        trg_BLD = self.self_attn_ln(trg_BLD + self.dropout(self_attn_BLD))
        # Encoder-Decoder Attetion
        enc_attn_BLD = self.encoder_attn(trg_BLD, enc_src_BCD, enc_src_BCD, src_mask_B11C)
        trg_BLD = self.enc_attn_ln(trg_BLD + self.dropout(enc_attn_BLD))
        ff_out_BLD = self.mlp(trg_BLD)
        trg_BLD = self.ff_ln(trg_BLD + self.dropout(ff_out_BLD))
        return trg_BLD


class MoEDecoderLayer(DecoderLayer):
    """A single layer of the decoder with Mixture of Experts in the feedforward layer."""

    def __init__(
        self,
        hid_dim: int,
        n_heads: int,
        dropout: float,
        attn_bias: bool,
        ff_mult: int,
        ff_activation: str,
        n_experts: int,
        top_k: int,
        capacity_factor: float,
    ) -> None:
        """Initializes the MoEDecoderLayer.

        Args:
            hid_dim: The hidden dimension size.
            n_heads: The number of attention heads.
            dropout: The dropout rate.
            attn_bias: Whether to use bias in the attention layers.
            ff_mult: The feed-forward expansion factor.
            ff_activation: The activation function type.
            n_experts: The number of experts in the MoE layer.
            top_k: The number of experts to use in the MoE layer.
            capacity_factor: The capacity factor for the MoE layer.
        """
        super().__init__(
            hid_dim=hid_dim,
            n_heads=n_heads,
            dropout=dropout,
            attn_bias=attn_bias,
            ff_mult=ff_mult,
            ff_activation=ff_activation,
        )
        # Override the MLP with MoE
        self.mlp = SparseMoE(
            hid_dim=hid_dim,
            n_experts=n_experts,
            top_k=top_k,
            ff_mult=ff_mult,
            ff_activation=ff_activation,
            dropout=dropout,
            capacity_factor=capacity_factor,
        )


class Decoder(nn.Module):
    """The decoder module.

    Shape suffixes convention:
        B: batch size
        C: the length of the input on which conditioning is done
           (in our case input_max_length)
        L: sequence length for decoder, in our case output_max_length
        D: model dimension (sometimes called d_model or embedding_dim)
        V: vocabulary size
    """

    def __init__(
        self,
        vocab_dim: int,
        hid_dim: int,
        context_window: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        attn_bias: bool,
        ff_mult: int,
        ff_activation: str,
    ) -> None:
        """Initializes the Decoder.

        Args:
            vocab_dim: The vocabulary size.
            hid_dim: The hidden dimension size.
            context_window: The context window size.
            n_layers: The number of decoder layers.
            n_heads: The number of attention heads.
            dropout: The dropout rate.
            attn_bias: Whether to use bias in the attention layers.
            ff_mult: The feed-forward expansion factor.
            ff_activation: The activation function type.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.tok_embedding = nn.Embedding(vocab_dim, hid_dim)
        self.pos_embedding = nn.Embedding(context_window, hid_dim)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    hid_dim=hid_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    attn_bias=attn_bias,
                    ff_mult=ff_mult,
                    ff_activation=ff_activation,
                )
                for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Linear(hid_dim, vocab_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))

    def forward(
        self,
        trg_BL: Tensor,
        enc_src_BCD: Tensor,
        src_mask_B11C: Tensor,
        trg_mask_B1LL: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the Decoder.

        Args:
            trg_BL: The target sequence tensor of shape (B, L).
            enc_src_BCD: The encoder output tensor of shape (B, C, D).
            src_mask_B11C: The source mask tensor of shape (B, 1, 1, C).
            trg_mask_B1LL: The target mask tensor of shape (B, 1, L, L).

        Returns:
            The output tensor of shape (B, L, V).
        """
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


class MoEDecoder(Decoder):
    """The decoder module with Mixture of Experts in the feedforward layers."""

    def __init__(
        self,
        vocab_dim: int,
        hid_dim: int,
        context_window: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        attn_bias: bool,
        ff_mult: int,
        ff_activation: str,
        n_experts: int,
        top_k: int,
        capacity_factor: float,
    ):
        """Initializes the MoEDecoder.

        Args:
            vocab_dim: The vocabulary size.
            hid_dim: The hidden dimension size.
            context_window: The context window size.
            n_layers: The number of decoder layers.
            n_heads: The number of attention heads.
            dropout: The dropout rate.
            attn_bias: Whether to use bias in the attention layers.
            ff_mult: The feed-forward expansion factor.
            ff_activation: The activation function type.
            n_experts: The number of experts in the MoE layer.
            top_k: The number of experts to use in the MoE layer.
            capacity_factor: The capacity factor for the MoE layer.
        """
        super().__init__(
            vocab_dim=vocab_dim,
            hid_dim=hid_dim,
            context_window=context_window,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            attn_bias=attn_bias,
            ff_mult=ff_mult,
            ff_activation=ff_activation,
        )
        # Override layers with MoE layers
        self.layers = nn.ModuleList(
            [
                MoEDecoderLayer(
                    hid_dim=hid_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    attn_bias=attn_bias,
                    ff_mult=ff_mult,
                    ff_activation=ff_activation,
                    n_experts=n_experts,
                    top_k=top_k,
                    capacity_factor=capacity_factor,
                )
                for _ in range(n_layers)
            ]
        )
