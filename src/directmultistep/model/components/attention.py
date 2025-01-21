from typing import cast

import torch
import torch.nn as nn

Tensor = torch.Tensor


class MultiHeadAttentionLayer(nn.Module):
    """
    Multi-head attention layer.

    This layer applies multi-head attention to the input tensors.

    Shape suffixes convention:
        B: batch size
        L: sequence length for decoder
        M: memory length (length of sequence being attended to)
        D: model dimension (sometimes called d_model or embedding_dim)
        H: number of attention heads in a layer

    Args:
        hid_dim: The hidden dimension size.
        n_heads: The number of attention heads.
        dropout: The dropout rate.
        attn_bias: Whether to use bias in the linear layers.
    """

    def __init__(
        self,
        hid_dim: int,
        n_heads: int,
        dropout: float,
        attn_bias: bool,
        # device: torch.device,
    ):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.query = nn.Linear(hid_dim, hid_dim, bias=attn_bias)
        self.key = nn.Linear(hid_dim, hid_dim, bias=attn_bias)
        self.value = nn.Linear(hid_dim, hid_dim, bias=attn_bias)

        self.projection = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        # self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(
        self,
        query_BLD: Tensor,
        key_BMD: Tensor,
        value_BMD: Tensor,
        mask_B11M: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass of the multi-head attention layer.

        Shape suffixes convention:
            B: batch size
            L: sequence length for decoder
            M: memory length (length of sequence being attended to)
            D: model dimension (sometimes called d_model or embedding_dim)
            H: number of attention heads in a layer

        Args:
            query_BLD: The query tensor of shape (B, L, D).
            key_BMD: The key tensor of shape (B, M, D).
            value_BMD: The value tensor of shape (B, M, D).
            mask_B11M: The attention mask of shape (B, 1, 1, M).

        Returns:
            The output tensor of shape (B, L, D).
        """
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
        attn_output_BLD = attn_output_BHLD.permute(0, 2, 1, 3).contiguous().view(B, L, self.hid_dim)
        output_BLD = cast(Tensor, self.projection(attn_output_BLD))
        return output_BLD
