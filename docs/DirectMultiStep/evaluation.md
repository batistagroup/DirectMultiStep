# Subset Evaluation

This documentation covers how to evaluate model performance on specific subsets of data using beam search.

## Example Use

Evaluating a model on a subset involves several steps:

1. Configure the evaluation parameters using `EvalConfig`
2. Load the model using `ModelFactory`
3. Initialize `ModelEvaluator` and run evaluation

See `use-examples/eval-subset.py` for a full example.

## Source Code

::: directmultistep.generation.eval
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - EvalConfig
        - ModelEvaluator
