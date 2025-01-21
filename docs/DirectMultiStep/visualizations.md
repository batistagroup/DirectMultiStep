# Visualizing Routes

## Example use

To visualize a path string, you can use the following snippet:

```python
from directmultistep.utils.web_visualize import draw_tree_from_path_string

path = "{'smiles':'O=C(c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1)N1CCN(CC2CC2)CC1','children':[{'smiles':'O=C(O)c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1','children':[{'smiles':'CCOC(=O)c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1','children':[{'smiles':'CCOC(=O)c1ccc(N)cc1'},{'smiles':'O=S(=O)(Cl)c1cccc2cccnc12'}]}]},{'smiles':'C1CN(CC2CC2)CCN1'}]}"
    
svg_str = draw_tree_from_path_string(
    path_string=path,
    save_path=Path("data/figures/desired_file_name"),
    width=400,
    height=400,
    x_margin=50,
    y_margin=100,
    theme="light",
)
```

## Source Code

::: directmultistep.utils.web_visualize
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - FilteredDict
        - ThemeType
        - ColorPalette
        - RetroSynthesisTree
        - TreeDimensions
        - compute_subtree_dimensions
        - compute_canvas_dimensions
        - check_overlap
        - draw_molecule
        - draw_tree_svg
        - create_tree_from_path_string
        - draw_tree_from_path_string
