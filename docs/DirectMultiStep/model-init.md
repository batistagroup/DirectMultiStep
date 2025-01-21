# Creating a model instance

There are several ways to create a DMS model instance, ranging from using preset configurations to custom configurations.

## Using Preset Configurations

The simplest way to create a model is using one of the preset configurations:

```py
from directmultistep.model import ModelFactory

factory = ModelFactory.from_preset("flash_10M", compile_model=True)
model = factory.create_model()
```

Available presets include: `deep_40M`, `explorer_xl_50M`, `flash_10M`, `flash_20M`, `flex_20M`, and `wide_40M`.

## Custom Configuration

For more control, you can create a custom configuration:

```python
from directmultistep.model.config import Seq2SeqConfig, EncoderAConfig, MoEDecoderConfig

config = Seq2SeqConfig(
    encoder=EncoderAConfig(
        vocab_dim=53,
        hid_dim=256,
        n_layers=6,
        n_heads=8,
        ff_mult=3,
        ff_activation="gelu",
        dropout=0.1,
        attn_bias=False,
        context_window=280,
        start_idx=0,
        mask_idx=51,
        pad_idx=52,
        initiate_steps=True,
        include_steps=True
    ),
    decoder=MoEDecoderConfig(
        vocab_dim=53,
        hid_dim=256,
        n_layers=6,
        n_heads=8,
        ff_mult=3,
        ff_activation="gelu",
        dropout=0.1,
        attn_bias=False,
        context_window=1075,
        start_idx=0,
        mask_idx=51,
        pad_idx=52,
        n_experts=3,
        top_k=2,
        capacity_factor=1.0,
    ),
)

factory = ModelFactory(config, device=None, compile_model=True)
model = factory.create_model()
```

## Configuration Types

The model supports different types of encoders and decoders:

- Encoders:
  - `EncoderAConfig`: EncoderA Type (the one we've been using so far)
  - `MoEEncoderConfig`: Mixture of Experts encoder

- Decoders:
  - `TransformerConfig`: Standard transformer decoder
  - `MoEDecoderConfig`: Mixture of Experts decoder

## Saving and Loading Configurations

Configurations can be saved to and loaded from YAML files:

```python
# Save configuration
config.save("model_config.yaml")

# Load configuration and create model
factory = ModelFactory.from_config_file("model_config.yaml")
model = factory.create_model()
```

## Source Code

::: directmultistep.model.config
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - TransformerConfig
        - MoEDecoderConfig
        - EncoderAConfig
        - MoEEncoderConfig
        - Seq2SeqConfig

::: directmultistep.model.factory
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - ModelFactory
