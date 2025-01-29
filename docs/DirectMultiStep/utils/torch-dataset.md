# Torch Dataset for Routes

This module provides a custom PyTorch Dataset class for handling reaction routes. It includes functionalities for tokenizing SMILES strings, reaction paths, and context information, as well as preparing data for training and generation.

## Example Use

`tokenize_path_string` is the most important function. It tokenizes a reaction path string. It uses a regular expression to split the string into tokens, and it can optionally add start-of-sequence (`<SOS>`) and end-of-sequence (`?`) tokens.

```python
from directmultistep.utils.dataset import tokenize_path_string

path_string = "{'smiles':'CC','children':[{'smiles':'CC(=O)O'}]}"
tokens = tokenize_path_string(path_string)
print(tokens)
```

## Notes on Path Start

In the `RoutesDataset` class, the `get_generation_with_sm` and `get_generation_no_sm` methods return an initial path tensor. This tensor is created from a `path_start` string, which is a partial path string that the model will start generating from. The `path_start` is `"{'smiles': 'product_smiles', 'children': [{'smiles':"`. The model will generate the rest of the path string from this starting point.

This design is important because a trained model always generates this `path_start` at the beginning of the sequence. By providing this as the initial input, we avoid wasting time generating this part and can focus on generating the rest of the reaction path.

The `prepare_input_tensors` function in `directmultistep.generate` allows for the provision of a custom `path_start` string. This is useful when you want to initiate the generation process from a specific point in the reaction path, instead of the default starting point. By modifying the `path_start` argument, you can control the initial state of the generation and explore different reaction pathways with user-defined intermediates.

## Source Code

::: directmultistep.utils.dataset
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - tokenize_smile
        - tokenize_smile_atom
        - tokenize_context
        - tokenize_path_string
        - RoutesDataset

::: directmultistep.generate
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - prepare_input_tensors