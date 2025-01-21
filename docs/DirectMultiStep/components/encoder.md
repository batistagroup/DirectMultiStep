# Encoder

This document describes the encoder components used in the DMS model.

## Base Encoder Layer

The basic building block of the encoder that processes input sequences.

### Components

#### **Self-Attention Block**

- Multi-head self-attention mechanism
- Layer normalization
- Residual connection

#### **Feed-Forward Block**

- Two-layer feed-forward network
- Configurable activation function (ReLU or GELU)
- Layer normalization
- Residual connection

## Source Code

::: directmultistep.model.components.encoder
    handler: python
    options:
      show_root_heading: true
      show_source: true
