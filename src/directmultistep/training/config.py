from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    # Data configs
    data_path: Path

    # Training setup
    run_name: str
    train_fname: str
    val_fname: str
    metadata_fname: str

    # Training hyperparameters
    batch_size: int
    learning_rate: float
    max_epochs: int

    # Scheduler configs
    warmup_steps: int
    decay_steps: int
    decay_factor: float

    pad_idx: int
    mask_idx: int

    # Checkpointing
    save_top_k: int = -1
    checkpoint_every_n_epochs: int = 2

    num_workers: int = 1
    n_devices: int = 1
    seed: int = 42

    accelerator: str = "auto"
    matmul_precision: str = "high"
    summary_depth: int = 2
    dist_strategy: str = "ddp_find_unused_parameters_true"

    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "value"

    def __post_init__(self) -> None:
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.run_name = f"{self.run_name}_seed={self.seed}"

        if self.matmul_precision not in ["high", "medium", "low"]:
            raise ValueError(f"{self.matmul_precision=} must be one of 'high', 'medium', or 'low'")

        if self.dist_strategy not in ["auto", "fsdp", "ddp", "ddp_spawn", "ddp_find_unused_parameters_true"]:
            raise ValueError(
                f"{self.dist_strategy=} must be one of 'fsdp', 'ddp', 'ddp_spawn', or 'ddp_find_unused_parameters_true'"
            )

        if self.gradient_clip_algorithm not in ["norm", "value"]:
            raise ValueError(f"{self.gradient_clip_algorithm=} must be one of 'norm' or 'value'")

    def save(self, path: Path) -> None:
        """Save config to YAML file.

        Args:
            path: Path to save config file
        """
        config_dict = asdict(self)
        config_dict["data_path"] = str(config_dict["data_path"])

        with open(path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        """Load config from YAML file.

        Args:
            path: Path to config file

        Returns:
            Loaded config object
        """
        with open(path) as f:
            config_dict = yaml.safe_load(f)

        config_dict["data_path"] = Path(config_dict["data_path"])
        instance = cls.__new__(cls)
        for key, value in config_dict.items():
            setattr(instance, key, value)
        return instance
