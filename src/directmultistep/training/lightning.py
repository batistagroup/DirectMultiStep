from typing import Any, Callable, cast

import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn

Tensor = torch.Tensor


def warmup_and_cosine_decay(warmup_steps: int, decay_steps: int, decay_factor: float) -> Callable[[int], float]:
    """Creates a learning rate schedule with warmup and cosine decay.

    The learning rate increases linearly during the warmup phase, then
    decreases following a cosine function during the decay phase, and
    finally remains constant at the decay factor.

    Args:
        warmup_steps: The number of steps for the warmup phase.
        decay_steps: The number of steps for the decay phase.
        decay_factor: The final learning rate factor after decay.

    Returns:
        A function that takes the current step as input and returns the
        corresponding learning rate factor.
    """

    def _get_new_lr(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        elif step >= warmup_steps and step < warmup_steps + decay_steps:
            factor = 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / decay_steps))
            return cast(float, max(factor, decay_factor))
        else:
            return decay_factor

    return _get_new_lr


class LTraining(pl.LightningModule):
    """A PyTorch Lightning module for training sequence-to-sequence models."""

    def __init__(
        self,
        pad_idx: int,
        mask_idx: int,
        lr: float,
        batch_size: int,
        warmup_steps: int = 4000,
        decay_steps: int = 24000,
        decay_factor: float = 0.1,
        model: nn.Module | None = None,
        criterion: nn.Module | None = None,
        processed_tokens: int = 0,
        start_idx: int = 0,
    ):
        """Initializes the PLTraining module.

        Args:
            pad_idx: The index of the padding token.
            mask_idx: The index of the mask token.
            lr: The initial learning rate.
            batch_size: The batch size.
            warmup_steps: The number of warmup steps for the learning rate scheduler.
            decay_steps: The number of decay steps for the learning rate scheduler.
            decay_factor: The decay factor for the learning rate scheduler.
            model: The sequence-to-sequence model.
            criterion: The loss function.
            processed_tokens: The number of tokens processed so far.
            start_idx: The index of the start token.
        """
        super().__init__()
        if model is not None:
            self.model = model
        if criterion is not None:
            self.criterion = criterion
        self.start_idx = start_idx
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.learning_rate = lr
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_factor = decay_factor
        self.processed_tokens = processed_tokens
        self.save_hyperparameters(ignore=["criterion", "model"])
        self.compute_loss = self.compute_loss_full

    def mask_src(self, src_BC: Tensor, masking_prob: float) -> Tensor:
        """Masks the source sequence with a given probability.

        Args:
            src_BC: The source sequence tensor of shape [B, C].
            masking_prob: The probability of masking a token.

        Returns:
            The masked source sequence tensor of shape [B, C].
        """
        mask_idx_BC = torch.rand(src_BC.shape).to(src_BC.device) < masking_prob
        not_pad_BC = src_BC != self.pad_idx
        final_mask_BC = mask_idx_BC & not_pad_BC
        masked_src_BC = src_BC.clone()
        masked_src_BC[final_mask_BC] = self.mask_idx
        return masked_src_BC

    def compute_loss_full(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Computes the loss for the full sequence training.

        This method calculates the loss for all tokens in the sequence.

        Args:
            batch: The input batch tensor.
            batch_idx: The index of the batch.

        Returns:
            The computed loss tensor.
        """
        src_item_BC = batch[0]
        tgt_item_BL = batch[1].long()
        steps_B1 = batch[2].view(-1, 1)
        masked_src_BC = self.mask_src(src_item_BC, masking_prob=0.05)
        # the output actually is [B, L-1, V] given slicing of tgt_item_BL
        output_BLV = self.model(masked_src_BC, tgt_item_BL[:, :-1], steps_B1)
        output_blV = output_BLV.view(-1, output_BLV.shape[-1])  # [B*(L-1), V]
        tgt_bl = tgt_item_BL[:, 1:].reshape(-1)  # [B*(L-1)]
        loss = self.criterion(output_blV, tgt_bl)
        self.processed_tokens += tgt_item_BL.shape[0] * tgt_item_BL.shape[1]
        return cast(Tensor, loss)

    def log_step_info(self, loss: Tensor, mode: str, prog_bar: bool) -> None:
        """Logs the loss and other training information.

        Args:
            loss: The loss tensor.
            mode: The mode of training ('train' or 'val').
            prog_bar: Whether to display the loss in the progress bar.
        """
        self.log(
            f"{mode}_loss",
            loss,
            batch_size=self.batch_size,
            prog_bar=prog_bar,
        )
        self.log("processed_tokens", self.processed_tokens)
        if mode == "train":
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log(f"{mode}_lr", current_lr, batch_size=self.batch_size)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Performs a single training step.

        Args:
            batch: The input batch tensor.
            batch_idx: The index of the batch.

        Returns:
            The computed loss tensor.
        """
        loss = self.compute_loss(batch, batch_idx)
        self.log_step_info(loss, "train", prog_bar=True)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Performs a single validation step.

        Args:
            batch: The input batch tensor.
            batch_idx: The index of the batch.

        Returns:
            The computed loss tensor.
        """
        loss = self.compute_loss(batch, batch_idx)
        self.log_step_info(loss, "val", prog_bar=True)
        return loss

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        """Configures the optimizer and learning rate scheduler.

        Returns:
            A tuple containing the list of optimizers and the list of
            learning rate schedulers.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # return optimizer
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup_and_cosine_decay(
                warmup_steps=self.warmup_steps,
                decay_steps=self.decay_steps,
                decay_factor=self.decay_factor,
            ),
            verbose=False,
        )
        lr_scheduler = {
            "scheduler": scheduler,  # The LR scheduler instance (required)
            "interval": "step",  # The unit of the scheduler's step size
            "frequency": 1,  # The frequency of the scheduler
        }
        return [optimizer], [lr_scheduler]

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Adds the processed tokens to the checkpoint.

        Args:
            checkpoint: The checkpoint dictionary.
        """
        # Add processed_tokens to the checkpoint dictionary
        checkpoint["processed_tokens"] = self.processed_tokens

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Loads the processed tokens from the checkpoint.

        Args:
            checkpoint: The checkpoint dictionary.
        """
        # Load processed_tokens from the checkpoint dictionary
        self.processed_tokens = checkpoint.get("processed_tokens", 0)
