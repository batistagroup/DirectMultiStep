from importlib import resources
from pathlib import Path

import torch
import torch.nn as nn
from torch import device as torch_device

from directmultistep.generation.eval import EvalConfig
from directmultistep.model.architecture import Seq2Seq
from directmultistep.model.components.decoder import Decoder, MoEDecoder
from directmultistep.model.components.encoder import Encoder, MoEEncoder
from directmultistep.model.config import (
    EncoderAConfig,
    MoEDecoderConfig,
    MoEEncoderConfig,
    Seq2SeqConfig,
    TransformerConfig,
)


class ModelFactory:
    """Factory class for creating and configuring models."""

    def __init__(
        self,
        config: Seq2SeqConfig,
        device: str | None = None,
        compile_model: bool = True,
        allow_mps: bool = False,
    ) -> None:
        """Initializes the ModelFactory.

        Args:
            config: The complete model configuration.
            device: Optional device specification. If None, the best available device is used.
            compile_model: Whether to compile the model using torch.compile.
            allow_mps: Whether to allow MPS device usage.
        """
        self.config = config
        self.device = self.determine_device(device, allow_mps)
        self.compile_model = compile_model

    def check_for_eval_config_updates(self, ec: EvalConfig) -> None:
        if isinstance(self.config.encoder, MoEEncoderConfig):
            if ec.enc_active_experts is None:
                raise ValueError("Encoder active experts must be set in eval config")
            self.config.encoder.top_k = ec.enc_active_experts
        if isinstance(self.config.decoder, MoEDecoderConfig):
            if ec.dec_active_experts is None:
                raise ValueError("Decoder active experts must be set in eval config")
            self.config.decoder.top_k = ec.dec_active_experts

    @staticmethod
    def determine_device(device: str | None = None, allow_mps: bool = False) -> torch_device:
        """Determines the appropriate device for model placement.

        Args:
            device: Optional device specification.

        Returns:
            The determined torch.device.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif allow_mps and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return torch.device(device)

    @staticmethod
    def _count_parameters(model: nn.Module) -> int:
        """Counts the trainable parameters in a model.

        Args:
            model: The PyTorch model.

        Returns:
            The number of trainable parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def create_model(self) -> Seq2Seq:
        """Creates and configures a Seq2Seq model based on the provided configuration.

        Returns:
            The configured Seq2Seq model.
        """
        # Create encoder based on configuration type
        if not isinstance(self.config.encoder, EncoderAConfig | MoEEncoderConfig):
            raise TypeError("Encoder config must be either EncoderAConfig or MoEEncoderConfig")
        if not isinstance(self.config.decoder, TransformerConfig | MoEDecoderConfig):
            raise TypeError("Decoder config must be either TransformerConfig or MoEDecoderConfig")

        encoder: Encoder | MoEEncoder
        if isinstance(self.config.encoder, MoEEncoderConfig):
            encoder = MoEEncoder(
                vocab_dim=self.config.encoder.vocab_dim,
                hid_dim=self.config.encoder.hid_dim,
                context_window=self.config.encoder.context_window,
                n_layers=self.config.encoder.n_layers,
                n_heads=self.config.encoder.n_heads,
                ff_mult=self.config.encoder.ff_mult,
                ff_activation=self.config.encoder.ff_activation,
                dropout=self.config.encoder.dropout,
                attn_bias=self.config.encoder.attn_bias,
                initiate_steps=self.config.encoder.initiate_steps,
                include_steps=self.config.encoder.include_steps,
                n_experts=self.config.encoder.n_experts,
                top_k=self.config.encoder.top_k,
                capacity_factor=self.config.encoder.capacity_factor,
            )
        else:
            encoder = Encoder(
                vocab_dim=self.config.encoder.vocab_dim,
                hid_dim=self.config.encoder.hid_dim,
                context_window=self.config.encoder.context_window,
                n_layers=self.config.encoder.n_layers,
                n_heads=self.config.encoder.n_heads,
                ff_mult=self.config.encoder.ff_mult,
                ff_activation=self.config.encoder.ff_activation,
                dropout=self.config.encoder.dropout,
                attn_bias=self.config.encoder.attn_bias,
                initiate_steps=self.config.encoder.initiate_steps,
                include_steps=self.config.encoder.include_steps,
            )

        decoder: Decoder | MoEDecoder
        if isinstance(self.config.decoder, MoEDecoderConfig):
            decoder = MoEDecoder(
                vocab_dim=self.config.decoder.vocab_dim,
                hid_dim=self.config.decoder.hid_dim,
                context_window=self.config.decoder.context_window,
                n_layers=self.config.decoder.n_layers,
                n_heads=self.config.decoder.n_heads,
                dropout=self.config.decoder.dropout,
                attn_bias=self.config.decoder.attn_bias,
                ff_mult=self.config.decoder.ff_mult,
                ff_activation=self.config.decoder.ff_activation,
                n_experts=self.config.decoder.n_experts,
                top_k=self.config.decoder.top_k,
                capacity_factor=self.config.decoder.capacity_factor,
            )
        else:
            decoder = Decoder(
                vocab_dim=self.config.decoder.vocab_dim,
                hid_dim=self.config.decoder.hid_dim,
                context_window=self.config.decoder.context_window,
                n_layers=self.config.decoder.n_layers,
                n_heads=self.config.decoder.n_heads,
                dropout=self.config.decoder.dropout,
                attn_bias=self.config.decoder.attn_bias,
                ff_mult=self.config.decoder.ff_mult,
                ff_activation=self.config.decoder.ff_activation,
            )

        model = Seq2Seq(
            encoder=encoder,
            decoder=decoder,
            src_pad_idx=self.config.encoder.pad_idx,
            trg_pad_idx=self.config.decoder.pad_idx,
        )

        model.to(self.device)

        if self.compile_model:
            model = torch.compile(model)  # type: ignore

        print(f"The model has {self._count_parameters(model):,} trainable parameters")
        return model

    @classmethod
    def from_config_file(
        cls,
        config_path: str | Path,
        device: str | None = None,
        compile_model: bool = True,
    ) -> "ModelFactory":
        """Creates a ModelFactory instance from a configuration file.

        Args:
            config_path: Path to the configuration file.
            device: Optional device specification.
            compile_model: Whether to compile the model.

        Returns:
            The configured ModelFactory instance.
        """
        config = Seq2SeqConfig.load(Path(config_path))
        return cls(config=config, device=device, compile_model=compile_model)

    @classmethod
    def from_preset(cls, preset_name: str, device: str | None = None, compile_model: bool = True) -> "ModelFactory":
        """Loads a preset configuration by name.

        Args:
            preset_name: The name of the preset configuration.
            device: Optional device specification.
            compile_model: Whether to compile the model.

        Returns:
            The configured ModelFactory instance.

        Raises:
            ValueError: If the preset is not found.
        """
        try:
            with resources.path("directmultistep.model.default_configs", f"{preset_name}.yaml") as config_path:
                return cls.from_config_file(config_path, device, compile_model)
        except FileNotFoundError as e:
            raise ValueError(
                f"Preset '{preset_name}' not found. Available presets: deep_40M, explorer_xl_50M, flash_10M, flash_20M, flex_20M, wide_40M"
            ) from e

    @staticmethod
    def load_checkpoint(model: Seq2Seq, ckpt_path: Path, device: torch.device) -> Seq2Seq:
        ckpt_torch = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt_torch)
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def load_lightning_checkpoint(model: Seq2Seq, ckpt_path: Path, device: torch.device) -> Seq2Seq:
        ckpt_lightning = torch.load(ckpt_path, map_location=device)
        ckpt_torch = {k.replace("model.", ""): v for k, v in ckpt_lightning["state_dict"].items()}
        model.load_state_dict(ckpt_torch)
        model.to(device)
        model.eval()
        return model


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
    factory = ModelFactory(config)
