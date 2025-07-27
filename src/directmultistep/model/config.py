from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class TransformerConfig:
    """Configuration for transformer components.

    Attributes:
        vocab_dim: Vocabulary dimension.
        hid_dim: Hidden dimension.
        n_layers: Number of layers.
        n_heads: Number of attention heads.
        ff_mult: Feedforward multiplier.
        ff_activation: Feedforward activation function ('gelu' or 'relu').
        dropout: Dropout probability.
        attn_bias: Whether to use attention bias.
        context_window: Context window size.
        start_idx: Start token index.
        mask_idx: Mask token index.
        pad_idx: Padding token index.
    """

    vocab_dim: int
    hid_dim: int
    n_layers: int
    n_heads: int
    ff_mult: int
    ff_activation: Literal["gelu", "relu"]
    dropout: float
    attn_bias: bool
    context_window: int
    start_idx: int
    mask_idx: int
    pad_idx: int

    def __post_init__(self) -> None:
        if self.hid_dim % self.n_heads != 0:
            raise ValueError(f"{self.hid_dim=} must be divisible by {self.n_heads=}")
        if self.ff_activation not in ["gelu", "relu"]:
            raise ValueError(f"{self.ff_activation=} must be either 'gelu' or 'relu'")

    def save(self, path: Path) -> None:
        """Save config to yaml file.

        Args:
            path: Path to save the config to.
        """
        data = asdict(self)
        data["model_type"] = self.__class__.__name__
        with open(path, "w") as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=False)

    @classmethod
    def load(cls: type[T], path: Path) -> T:
        """Load config from yaml file.

        Args:
            path: Path to load the config from.

        Returns:
            Loaded config.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


@dataclass
class MoEDecoderConfig(TransformerConfig):
    """Configuration for Mixture of Experts decoder components.

    Attributes:
        n_experts: Number of experts.
        top_k: Number of experts to use in forward pass.
        capacity_factor: Capacity factor for experts.
    """

    n_experts: int
    top_k: int
    capacity_factor: float


@dataclass
class EncoderAConfig(TransformerConfig):
    """Configuration for EncoderA components.

    Attributes:
        initiate_steps: Whether to initiate steps.
        include_steps: Whether to include steps.
    """

    initiate_steps: bool
    include_steps: bool


@dataclass
class MoEEncoderConfig(EncoderAConfig):
    """Configuration for Mixture of Experts encoder components.

    Attributes:
        n_experts: Number of experts.
        top_k: Number of experts to use in forward pass.
        capacity_factor: Capacity factor for experts.
    """

    n_experts: int
    top_k: int
    capacity_factor: float


@dataclass
class Seq2SeqConfig:
    """Complete model configuration.

    Attributes:
        encoder: Encoder configuration.
        decoder: Decoder configuration.
    """

    encoder: TransformerConfig
    decoder: TransformerConfig

    def save(self, path: Path) -> None:
        """Save config to yaml file.

        Args:
            path: Path to save the config to.
        """
        config_dict = {
            "encoder": asdict(self.encoder) | {"model_type": self.encoder.__class__.__name__},
            "decoder": asdict(self.decoder) | {"model_type": self.decoder.__class__.__name__},
        }
        with open(path, "w") as f:
            yaml.dump(config_dict, f, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> "Seq2SeqConfig":
        """Load config from yaml file.

        Args:
            path: Path to load the config from.

        Returns:
            Loaded Seq2SeqConfig.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        # Determine correct encoder/decoder types based on model_type
        encoder_data = data.pop("encoder")
        decoder_data = data.pop("decoder")

        model_type_to_config = {
            "TransformerConfig": TransformerConfig,
            "MoEDecoderConfig": MoEDecoderConfig,
            "EncoderAConfig": EncoderAConfig,
            "MoEEncoderConfig": MoEEncoderConfig,
        }

        encoder_model_type = encoder_data.pop("model_type")
        decoder_model_type = decoder_data.pop("model_type")

        encoder_type = model_type_to_config[encoder_model_type]
        decoder_type = model_type_to_config[decoder_model_type]

        encoder = encoder_type(**encoder_data)
        decoder = decoder_type(**decoder_data)

        return cls(encoder=encoder, decoder=decoder, **data)


if __name__ == "__main__":
    config = Seq2SeqConfig(
        encoder=TransformerConfig(
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
