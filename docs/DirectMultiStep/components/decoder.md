# Decoder

This document describes the decoder components used in the DMS model.

## Base Decoder Layer

The basic building block of the decoder that processes target sequences.

### Components

#### **Self-Attention Block**

- Multi-head self-attention mechanism
- Causal masking to prevent looking ahead
- Layer normalization
- Residual connection

#### **Cross-Attention Block**

- Multi-head attention over encoder outputs
- Allows decoder to focus on relevant input parts
- Layer normalization
- Residual connection

#### **Feed-Forward Block**

- Two-layer feed-forward network
- Configurable activation function (ReLU or GELU)
- Layer normalization
- Residual connection

## Source Code

::: directmultistep.model.components.decoder
    handler: python
    options:
      show_root_heading: true
      show_source: true
