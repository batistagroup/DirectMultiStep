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

import torch
import lightning as L
from pathlib import Path
from DirectMultiStep.Models.Configure import (
    prepare_model,
    determine_device,
    VanillaTransformerConfig,
)
from DirectMultiStep.Models.Training import PLTraining
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichModelSummary
import DirectMultiStep.helpers as helpers

data_path = Path(__file__).resolve().parent / "Data" / "Processed"
train_path = Path(__file__).resolve().parent / "Data" / "Training"
run_name = "nosm_run_name"
batch_size = 32
lr = 3e-4
steps_per_epoch = 30299
max_epochs = 4
L.seed_everything(42)
n_devices = 4
torch.set_float32_matmul_precision("high")

model_10m = dict(n_layers=6, ff_mult=3, hid_dim=256)
model_60m = dict(n_layers=8, ff_mult=4, hid_dim=512)

model_config = VanillaTransformerConfig(
    input_dim=53,
    output_dim=53,
    input_max_length=145,
    output_max_length=1074 + 1,
    pad_index=52,
    attn_bias=False,
    ff_activation="gelu",
    **model_10m,
    # **model_60m,
)

# enc and dec configs may be different
model = prepare_model(enc_config=model_config, dec_config=model_config)
if __name__ == "__main__":
    # Training hyperparameters
    mask_idx, pad_idx = 51, 52

    ds_train, ds_val = helpers.prepare_datasets_nosm(
        train_data_path=data_path / "all_dataset_nperms=3_nosm.pkl",
        val_data_path=data_path / "n1_dataset_nperms=1_nosm.pkl",
        metadata_path=data_path / "character_dictionary.yaml",
    )
    dl_train = torch.utils.data.DataLoader(
        dataset=ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=120,
        pin_memory=True,
    )
    dl_val = torch.utils.data.DataLoader(
        dataset=ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=120,
        pin_memory=True,
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="mean")
    lightning_model = PLTraining(
        model=model,
        pad_idx=pad_idx,
        mask_idx=mask_idx,
        criterion=criterion,
        lr=lr,
        batch_size=batch_size,
        warmup_steps=int(steps_per_epoch * 0.1),
        decay_steps=steps_per_epoch * 20,
        decay_factor=0.1,
    )

    device = determine_device()
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath=train_path / run_name, save_last=True, save_top_k=1
    )
    model_summary = RichModelSummary(max_depth=2)

    trainer = L.Trainer(
        default_root_dir=train_path / run_name,
        max_epochs=max_epochs,
        accelerator=device,
        devices=n_devices,
        num_nodes=1,
        strategy="fsdp",  # if using CUDA
        callbacks=[checkpoint_callback, model_summary],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="value",
    )
    latest_ckpt = helpers.find_checkpoint(train_path, run_name)
    if latest_ckpt is not None:
        print(f"Loading model from {latest_ckpt}")
        trainer.fit(lightning_model, dl_train, dl_val, ckpt_path=latest_ckpt)
    else:
        trainer.fit(lightning_model, dl_train, dl_val)
