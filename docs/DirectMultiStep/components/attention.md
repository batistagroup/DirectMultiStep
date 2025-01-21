# Attention

This document describes the attention mechanisms used in the DMS model.

## Summary

The core mechanism of attention emerges from needing to selectively focus on relevant information while processing sequences. When encoding tokens, each position must consider its relationship with all others to capture context. Attention computes similarity scores between each query position and all possible key positions, essentially asking "how relevant is each key to my current query?" These raw similarity scores are normalized through softmax to produce attention weights that sum to 1, creating a probability distribution over the keys for each query. The weighted sum of values according to these attention weights produces the final attention output, allowing the model to synthesize information from multiple positions with varying degrees of influence.

### Flash Attention

Flash Attention reformulates attention computation to maximize use of fast SRAM cache while minimizing slower DRAM memory access. Rather than computing and storing the full attention matrix at once, it splits the computation into smaller blocks that fit in SRAM, computing partial attention scores and incrementally aggregating them. This tiling approach, combined with local softmax normalization within blocks, achieves mathematically equivalent results while drastically reducing memory bandwidth requirements. The key insight is maintaining rolling statistics of softmax normalization terms across blocks, allowing processing of long sequences without materializing the full attention matrix in memory – trading increased computation for reduced memory usage, which is favorable on modern hardware where memory bandwidth often constrains performance more than computational capacity.

### Shape Convention

The shape suffixes follow a consistent convention:

- `B`: Batch size
- `L`: Target sequence length
- `M`: Memory/source sequence length
- `D`: Model hidden dimension
- `H`: Number of attention heads

## Source Code

::: directmultistep.model.components.attention
    handler: python
    members: MultiHeadAttentionLayer
    options:
      show_root_heading: true
      show_source: true
