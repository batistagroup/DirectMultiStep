# Monitoring Training

This guide explains how to monitor and visualize training progress for DMS models.

## Basic Usage

The simplest way to visualize training progress is using the provided plotting utilities in `use-examples/visualize_train_curves.py`

## Run Configuration

Use `RunConfig` to specify which training runs to visualize:

```python
from directmultistep.analysis.training import RunConfig

run = RunConfig(
    run_name="flash_10M",      # Folder name of the run
    trace_name="Flash Model",  # Display name for the traces
    include_val=True          # Whether to include validation curve
)
```

## Training Curves

The `plot_training_curves` function creates a figure showing:

- Training loss curves (solid lines)
- Validation loss curves (dotted lines with markers)
- X-axis shows number of processed tokens
- Hovering over validation points shows epoch information

## Learning Rate Curves

The `plot_learning_rates` function visualizes the learning rate schedule:

- Shows learning rate vs. training step
- Useful for verifying learning rate schedules
- Multiple runs can be compared on the same plot

## Advanced Usage

For more control over visualization, you can load the training data directly:

```python
from directmultistep.analysis.training import load_training_df

# Load training data
df = load_training_df(train_path, "flash_10M")

# Ignore specific training runs by ID
df = load_training_df(train_path, "flash_10M", ignore_ids=[0, 1])
```

The returned DataFrame contains columns:

- `processed_tokens`: Number of tokens processed
- `train_loss`: Training loss
- `val_loss`: Validation loss (if available)
- `train_lr`: Learning rate
- `epoch`: Current epoch
- Additional metrics depending on the training configuration

## Source Code

::: directmultistep.analysis.training
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - RunConfig
        - plot_training_curves
        - plot_learning_rates
        - load_training_df
