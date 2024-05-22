# MIT License

# Copyright (c) 2024 Batista Lab (Yale University)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import lightning as pl
import numpy as np
from typing import Callable, Optional
import torch
import torch.nn as nn
from .Architecture import Seq2Seq

Tensor = torch.Tensor


def _warmup_and_cosine_decay(
    warmup_steps: int, decay_steps: int, decay_factor: float
) -> Callable:
    def _get_new_lr(step):
        if step < warmup_steps:
            return step / warmup_steps
        elif step >= warmup_steps and step < warmup_steps + decay_steps:
            factor = 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / decay_steps))
            return max(factor, decay_factor)
        else:
            return decay_factor

    return _get_new_lr


class PLTraining(pl.LightningModule):
    def __init__(
        self,
        pad_idx: int,
        mask_idx: int,
        lr: float,
        batch_size: int,
        warmup_steps: int = 4000,
        decay_steps: int = 24000,
        decay_factor: float = 0.1,
        model: Optional[Seq2Seq] = None,
        criterion: Optional[nn.Module] = None,
    ):
        super().__init__()
        if model is not None:
            self.model = model
        if criterion is not None:
            self.criterion = criterion
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.learning_rate = lr
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_factor = decay_factor
        self.save_hyperparameters(ignore=["criterion", "model"])

    def mask_src(self, src_BC: Tensor, masking_prob: float) -> Tensor:
        mask_idx_BC = torch.rand(src_BC.shape).to(src_BC.device) < masking_prob
        not_pad_BC = src_BC != self.pad_idx
        final_mask_BC = mask_idx_BC & not_pad_BC
        masked_src_BC = src_BC.clone()
        masked_src_BC[final_mask_BC] = self.mask_idx
        return masked_src_BC

    def compute_loss(self, batch, batch_idx):
        """
        enc_item - product_item + one_sm_item
        dec_item - path_string
        steps - number of steps in the route
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
        return loss

    def log_step_info(self, loss, mode: str, prog_bar: bool):
        self.log(
            f"{mode}_loss",
            loss,
            batch_size=self.batch_size,
            prog_bar=prog_bar,
            sync_dist=True,
        )
        if mode == "train":
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log(
                f"{mode}_lr", current_lr, batch_size=self.batch_size, sync_dist=True
            )

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log_step_info(loss, "train", prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log_step_info(loss, "val", prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # return optimizer
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=_warmup_and_cosine_decay(
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
