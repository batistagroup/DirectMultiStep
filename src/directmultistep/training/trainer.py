from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from torch.utils.data import DataLoader

from directmultistep import helpers
from directmultistep.training.config import TrainingConfig
from directmultistep.training.lightning import LTraining
from directmultistep.utils.dataset import RoutesDataset

Tensor = torch.Tensor


class ModelTrainer:
    """High-level trainer class that orchestrates the training process."""

    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration.

        Args:
            config: Training configuration
        """
        self.config = config
        self._setup_environment()

    def _setup_environment(self) -> None:
        """Configure training environment."""
        L.seed_everything(self.config.seed)
        torch.set_float32_matmul_precision(self.config.matmul_precision)

    def _create_lightning_module(self, model: torch.nn.Module) -> LTraining:
        """Create the Lightning training module.

        Args:
            model: The model to train

        Returns:
            Configured PLTraining module
        """
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_idx, reduction="mean")

        return LTraining(
            model=model,
            pad_idx=self.config.pad_idx,
            mask_idx=self.config.mask_idx,
            criterion=criterion,
            lr=self.config.learning_rate,
            batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            decay_steps=self.config.decay_steps,
            decay_factor=self.config.decay_factor,
        )

    def _setup_callbacks(self) -> list[Any]:
        """Configure training callbacks.

        Returns:
            List of Lightning callbacks
        """
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=self.config.data_path / "training" / self.config.run_name,
            save_last=True,
            save_top_k=self.config.save_top_k,
            every_n_epochs=self.config.checkpoint_every_n_epochs,
        )

        return [checkpoint_callback, RichModelSummary(max_depth=self.config.summary_depth)]

    def _create_trainer(self) -> L.Trainer:
        """Create Lightning trainer.

        Returns:
            Configured Lightning trainer
        """
        return L.Trainer(
            default_root_dir=self.config.data_path / "training" / self.config.run_name,
            max_epochs=self.config.max_epochs,
            accelerator=self.config.accelerator,
            devices=self.config.n_devices,
            num_nodes=1,
            strategy=self.config.dist_strategy,
            callbacks=self._setup_callbacks(),
            gradient_clip_val=self.config.gradient_clip_val,
            gradient_clip_algorithm=self.config.gradient_clip_algorithm,
        )

    def _create_dataloaders(
        self,
        train_dataset: RoutesDataset,
        val_dataset: RoutesDataset,
    ) -> tuple[DataLoader[tuple[Tensor, ...]], DataLoader[tuple[Tensor, ...]]]:
        """Create training and validation dataloaders.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset

        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

        return train_loader, val_loader

    def train(
        self,
        model: torch.nn.Module,
        train_dataset: RoutesDataset,
        val_dataset: RoutesDataset,
    ) -> None:
        """Train the model.

        Args:
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            checkpoint_path: Optional path to checkpoint for resuming training
        """
        lightning_model = self._create_lightning_module(model)
        trainer = self._create_trainer()
        dl_train, dl_val = self._create_dataloaders(train_dataset, val_dataset)
        latest_ckpt = helpers.find_checkpoint(self.config.data_path / "training", self.config.run_name)

        if latest_ckpt is not None:
            print(f"Loading model from {latest_ckpt}")
            trainer.fit(lightning_model, dl_train, dl_val, ckpt_path=latest_ckpt)
        else:
            trainer.fit(lightning_model, dl_train, dl_val)
