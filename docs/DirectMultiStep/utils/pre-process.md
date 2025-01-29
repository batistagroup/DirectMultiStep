# Multistep Route Pre-processing

This module provides useful data structure classes and helper functions for preprocessing multistep routes for training and testing DirectMultiStep models.

## Example Use

The most frequently used data structure is `FilteredDict`, a dictionary format for multistep routes used in DirectMultiStep models. Several useful functions are available, such as `canonicalize_smiles`, `max_tree_depth`, `find_leaves`, `stringify_dict`, and `generate_permutations`, among others. For example:

```python
from directmultistep.utils.pre_process import stringify_dict

path_string = "{'smiles':'CNCc1cc(-c2ccccc2F)n(S(=O)(=O)c2cccnc2)c1','children':[{'smiles':'O=Cc1cc(-c2ccccc2F)n(S(=O)(=O)c2cccnc2)c1','children':[{'smiles':'O=Cc1c[nH]c(-c2ccccc2F)c1'},{'smiles':'O=S(=O)(Cl)c1cccnc1'}]},{'smiles':'CN'}]}"

# This should evaluate to True, as it compares the stringified version of your FilteredDict
print(stringify_dict(eval(path_string)) == path_string)
```

## Source Code

::: directmultistep.utils.pre_process
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - PaRoutesDict
        - FilteredDict
        - filter_mol_nodes
        - max_tree_depth
        - find_leaves
        - canonicalize_smiles
        - stringify_dict
        - generate_permutations
        - is_convergent
