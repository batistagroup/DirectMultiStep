# Training

## Example Use

Training a model involves three main steps:

1. Create a model configuration and instance using `ModelFactory`
2. Configure the training parameters using `TrainingConfig`
3. Initialize the `ModelTrainer` and start training

See `use-examples/train_model.py` for a full example.

## Source Code

::: directmultistep.training.config
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - TrainingConfig

::: directmultistep.training.trainer
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - ModelTrainer

::: directmultistep.training.lightning
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - warmup_and_cosine_decay
        - LTraining
