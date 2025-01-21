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


class EncoderLayer(nn.Module):
    """A single layer of the encoder.

    Shape suffixes convention:
        B: batch size
        C: the length of the input on which conditioning is done
           (in our case input_max_length)
        D: model dimension (sometimes called d_model or embedding_dim)
    """

    def __init__(
        self,
        hid_dim: int,
        n_heads: int,
        ff_mult: int,
        ff_activation: str,
        dropout: float,
        attn_bias: bool,
    ):
        """Initializes the EncoderLayer.

        Args:
            hid_dim: The hidden dimension size.
            n_heads: The number of attention heads.
            ff_mult: The feed-forward expansion factor.
            ff_activation: The activation function type.
            dropout: The dropout rate.
            attn_bias: Whether to use bias in the attention layers.
        """
        super().__init__()

        self.attn_ln = nn.LayerNorm(hid_dim)
        self.ff_ln = nn.LayerNorm(hid_dim)
        self.attention = MultiHeadAttentionLayer(
            hid_dim=hid_dim,
            n_heads=n_heads,
            dropout=dropout,
            attn_bias=attn_bias,
        )
        self.mlp = PositionwiseFeedforwardLayer(
            hid_dim=hid_dim,
            ff_mult=ff_mult,
            ff_activation=activation_dict[ff_activation],
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_BCD: Tensor, src_mask_B11C: Tensor) -> Tensor:
        """Forward pass of the EncoderLayer.

        Args:
            input_BCD: The input tensor of shape (B, C, D).
            src_mask_B11C: The source mask tensor of shape (B, 1, 1, C).

        Returns:
            The output tensor of shape (B, C, D).
        """
        attn_output_BCD = self.attention(input_BCD, input_BCD, input_BCD, src_mask_B11C)
        src_BCD = self.attn_ln(input_BCD + self.dropout(attn_output_BCD))
        ff_out_BCD = self.mlp(src_BCD)
        final_out_BLD = self.ff_ln(src_BCD + self.dropout(ff_out_BCD))
        return cast(Tensor, final_out_BLD)


class MoEEncoderLayer(nn.Module):
    """A single layer of the MoE encoder.

    Shape suffixes convention:
        B: batch size
        C: the length of the input on which conditioning is done
           (in our case input_max_length)
        D: model dimension (sometimes called d_model or embedding_dim)
    """

    def __init__(
        self,
        hid_dim: int,
        n_heads: int,
        n_experts: int,
        top_k: int,
        ff_mult: int,
        ff_activation: str,
        dropout: float,
        attn_bias: bool,
        capacity_factor: float,
    ):
        """Initializes the MoEEncoderLayer.

        Args:
            hid_dim: The hidden dimension size.
            n_heads: The number of attention heads.
            n_experts: The number of experts in the MoE layer.
            top_k: The number of experts to use in the MoE layer.
            ff_mult: The feed-forward expansion factor.
            ff_activation: The activation function type.
            dropout: The dropout rate.
            attn_bias: Whether to use bias in the attention layers.
            capacity_factor: The capacity factor for the MoE layer.
        """
        super().__init__()

        self.attn_ln = nn.LayerNorm(hid_dim)
        self.ff_ln = nn.LayerNorm(hid_dim)
        self.attention = MultiHeadAttentionLayer(
            hid_dim=hid_dim,
            n_heads=n_heads,
            dropout=dropout,
            attn_bias=attn_bias,
        )
        self.mlp = SparseMoE(
            hid_dim=hid_dim,
            n_experts=n_experts,
            top_k=top_k,
            ff_mult=ff_mult,
            ff_activation=ff_activation,
            dropout=dropout,
            capacity_factor=capacity_factor,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_BCD: Tensor, src_mask_B11C: Tensor) -> Tensor:
        """Forward pass of the MoEEncoderLayer.

        Args:
            input_BCD: The input tensor of shape (B, C, D).
            src_mask_B11C: The source mask tensor of shape (B, 1, 1, C).

        Returns:
            The output tensor of shape (B, C, D).
        """
        attn_output_BCD = self.attention(input_BCD, input_BCD, input_BCD, src_mask_B11C)
        src_BCD = self.attn_ln(input_BCD + self.dropout(attn_output_BCD))
        ff_out_BCD = self.mlp(src_BCD)
        final_out_BLD = self.ff_ln(src_BCD + self.dropout(ff_out_BCD))
        return cast(Tensor, final_out_BLD)


class Encoder(nn.Module):
    """The encoder module.

    Shape suffixes convention:
        B: batch size
        C: the length of the input on which conditioning is done
           (in our case input_max_length)
        D: model dimension (sometimes called d_model or embedding_dim)
    """

    def __init__(
        self,
        vocab_dim: int,
        hid_dim: int,
        context_window: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int,
        ff_activation: str,
        dropout: float,
        attn_bias: bool,
        initiate_steps: bool,
        include_steps: bool,
    ):
        """Initializes the Encoder.

        Args:
            vocab_dim: The vocabulary dimension size.
            hid_dim: The hidden dimension size.
            context_window: The context window size.
            n_layers: The number of encoder layers.
            n_heads: The number of attention heads.
            ff_mult: The feed-forward expansion factor.
            ff_activation: The activation function type.
            dropout: The dropout rate.
            attn_bias: Whether to use bias in the attention layers.
            initiate_steps: Whether to initiate step embeddings.
            include_steps: Whether to include step embeddings.
        """
        super().__init__()

        self.tok_embedding = nn.Embedding(vocab_dim, hid_dim)
        self.pos_embedding = nn.Embedding(context_window, hid_dim)
        if initiate_steps:
            self.step_embedding = nn.Embedding(1, hid_dim)
        self.include_steps = include_steps

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hid_dim=hid_dim,
                    n_heads=n_heads,
                    ff_mult=ff_mult,
                    ff_activation=ff_activation,
                    dropout=dropout,
                    attn_bias=attn_bias,
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))

    def forward(self, src_BC: Tensor, src_mask_B11C: Tensor, steps_B1: Tensor) -> Tensor:
        """Forward pass of the Encoder.

        Args:
            src_BC: The source input tensor of shape (B, C).
            src_mask_B11C: The source mask tensor of shape (B, 1, 1, C).
            steps_B1: The step tensor of shape (B, 1).

        Returns:
            The output tensor of shape (B, C, D).
        """
        B, C = src_BC.shape
        tok_emb_BCD = self.tok_embedding(src_BC) * self.scale.to(src_BC)
        # below [C] -> [1, C] -> [B, C]
        pos_BC = torch.arange(0, C).unsqueeze(0).repeat(B, 1).to(src_BC)
        pos_emb_BCD = self.pos_embedding(pos_BC)
        comb_BCD = tok_emb_BCD + pos_emb_BCD
        if self.include_steps:
            # [C] -> [1, C] -> [B, C]
            step_BC = torch.zeros(C).unsqueeze(0).repeat(B, 1).long().to(src_BC)
            step_emb_BCD = self.step_embedding(step_BC) * steps_B1.view(-1, 1, 1)
            comb_BCD += step_emb_BCD
        src_BCD = self.dropout(comb_BCD)
        for layer in self.layers:
            src_BCD = layer(src_BCD, src_mask_B11C)
        return cast(Tensor, src_BCD)


class MoEEncoder(nn.Module):
    """The MoE encoder module.

    Shape suffixes convention:
        B: batch size
        C: the length of the input on which conditioning is done
           (in our case input_max_length)
        D: model dimension (sometimes called d_model or embedding_dim)
    """

    def __init__(
        self,
        vocab_dim: int,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        n_experts: int,
        top_k: int,
        ff_mult: int,
        ff_activation: str,
        dropout: float,
        attn_bias: bool,
        context_window: int,
        initiate_steps: bool,
        include_steps: bool,
        capacity_factor: float,
    ):
        """Initializes the MoEEncoder.

        Args:
            vocab_dim: The vocabulary dimension size.
            hid_dim: The hidden dimension size.
            n_layers: The number of encoder layers.
            n_heads: The number of attention heads.
            n_experts: The number of experts in the MoE layer.
            top_k: The number of experts to use in the MoE layer.
            ff_mult: The feed-forward expansion factor.
            ff_activation: The activation function type.
            dropout: The dropout rate.
            attn_bias: Whether to use bias in the attention layers.
            context_window: The context window size.
            initiate_steps: Whether to initiate step embeddings.
            include_steps: Whether to include step embeddings.
            capacity_factor: The capacity factor for the MoE layer.
        """
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_dim, hid_dim)
        self.pos_embedding = nn.Embedding(context_window, hid_dim)
        if initiate_steps:
            self.step_embedding = nn.Embedding(1, hid_dim)
        self.include_steps = include_steps

        self.layers = nn.ModuleList(
            [
                MoEEncoderLayer(
                    hid_dim=hid_dim,
                    n_heads=n_heads,
                    n_experts=n_experts,
                    top_k=top_k,
                    ff_mult=ff_mult,
                    ff_activation=ff_activation,
                    dropout=dropout,
                    attn_bias=attn_bias,
                    capacity_factor=capacity_factor,
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))

    def forward(self, src_BC: Tensor, src_mask_B11C: Tensor, steps_B1: Tensor) -> Tensor:
        """Forward pass of the MoEEncoder.

        Args:
            src_BC: The source input tensor of shape (B, C).
            src_mask_B11C: The source mask tensor of shape (B, 1, 1, C).
            steps_B1: The step tensor of shape (B, 1).

        Returns:
            The output tensor of shape (B, C, D).
        """
        B, C = src_BC.shape
        tok_emb_BCD = self.tok_embedding(src_BC) * self.scale.to(src_BC)
        # below [C] -> [1, C] -> [B, C]
        pos_BC = torch.arange(0, C).unsqueeze(0).repeat(B, 1).to(src_BC)
        pos_emb_BCD = self.pos_embedding(pos_BC)
        comb_BCD = tok_emb_BCD + pos_emb_BCD
        if self.include_steps:
            # [C] -> [1, C] -> [B, C]
            step_BC = torch.zeros(C).unsqueeze(0).repeat(B, 1).long().to(src_BC)
            step_emb_BCD = self.step_embedding(step_BC) * steps_B1.view(-1, 1, 1)
            comb_BCD += step_emb_BCD
        src_BCD = self.dropout(comb_BCD)
        for layer in self.layers:
            src_BCD = layer(src_BCD, src_mask_B11C)
        return cast(Tensor, src_BCD)
