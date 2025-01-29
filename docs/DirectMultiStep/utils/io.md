# Input/Output Utilities

This module provides functions for loading and saving datasets, as well as converting between different data formats. It is useful for preparing data for training and testing DirectMultiStep models.

## Example Use

The most useful functions are `load_dataset_sm`, `load_dataset_nosm`, `save_dataset_sm`, and `load_pharma_compounds`. These functions allow you to load and save datasets in a variety of formats.

```python
from pathlib import Path
from directmultistep.utils.io import load_pharma_compounds

data_path = Path.cwd() / "data"

_products, _sms, _path_strings, _steps_list, nameToIdx = load_pharma_compounds(data_path / "pharma_compounds.json")
```

## Source Code

::: directmultistep.utils.io
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - DatasetDict
        - load_dataset_sm
        - load_dataset_nosm
        - save_dataset_sm
        - convert_dict_of_lists_to_list_of_dicts
        - convert_list_of_dicts_to_dict_of_lists
        - load_pharma_compounds
        - load_commercial_stock