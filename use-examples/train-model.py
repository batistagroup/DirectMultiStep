from pathlib import Path

from directmultistep import helpers
from directmultistep.model import ModelFactory
from directmultistep.training import ModelTrainer, TrainingConfig

__mode__ = "local"
assert __mode__ in ["local", "cluster"]

if __mode__ == "local":
    # replace with .parent if you place train-model.py in root folder
    # .parents[1] if you keep it in use-examples
    base_path = Path(__file__).resolve().parents[1]
    n_workers = 1
    n_devices = 1
    accelerator = "cpu"
    batch_size = 8
elif __mode__ == "shared":
    base_path = Path(__file__).resolve().parent  # change it to your path
    n_workers = 64
    n_devices = 1
    accelerator = "auto"
    batch_size = 32 * 4
data_path = base_path / "data"

factory = ModelFactory.from_preset("flash_10M", compile_model=False)
# or any other preset name from src/directmultistep/dms/model/default_configs
# or create your own config, see src/directmultistep/model/factory.py for examples
config = TrainingConfig(
    data_path=data_path,
    run_name="van_sm_6x3_6x3_256_noboth",
    train_fname="unique_dataset_nperms=3_nsms=all_noboth_train=0.95.pkl",
    val_fname="unique_dataset_nperms=3_nsms=all_noboth_val=0.05.pkl",
    metadata_fname="dms_dictionary.yaml",
    batch_size=batch_size,
    learning_rate=2e-4,
    max_epochs=40,
    warmup_steps=3000,
    decay_steps=80_000,
    decay_factor=0.1,
    pad_idx=factory.config.decoder.pad_idx,
    mask_idx=factory.config.decoder.mask_idx,
    save_top_k=-1,  # -1 will save all
    checkpoint_every_n_epochs=2,  # every 2 epochs
    summary_depth=2,
    accelerator=accelerator,
    num_workers=n_workers,
    n_devices=n_devices,
)

# Save configs to logbook
logbook_path = data_path / "configs" / "logbook" / config.run_name
logbook_path.mkdir(parents=True, exist_ok=True)

config.save(logbook_path / "training_config.yaml")
factory.config.save(logbook_path / "model_config.yaml")


train_dataset, val_dataset = helpers.prepare_datasets(
    train_data_path=data_path / "processed" / config.train_fname,
    val_data_path=data_path / "processed" / config.val_fname,
    metadata_path=data_path / "configs" / config.metadata_fname,
    load_sm=True,
    mode="training",
)

model = factory.create_model()
trainer = ModelTrainer(config)

if __name__ == "__main__":
    # this has to be in main block to avoid issues with multiprocessing
    trainer.train(model, train_dataset, val_dataset)
