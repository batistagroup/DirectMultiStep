from DirectMultiStep.Utils.Visualize import draw_tree_from_path_string
from pathlib import Path

data_path = Path(__file__).resolve().parent / "Data"
fig_path = data_path / "Figures"

if __name__ == "__main__":
    path = "{'smiles':'O=C(c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1)N1CCN(CC2CC2)CC1','children':[{'smiles':'O=C(O)c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1','children':[{'smiles':'CCOC(=O)c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1','children':[{'smiles':'CCOC(=O)c1ccc(N)cc1'},{'smiles':'O=S(=O)(Cl)c1cccc2cccnc12'}]}]},{'smiles':'C1CN(CC2CC2)CCN1'}]}"
    draw_tree_from_path_string(
        path_string=path, save_path=fig_path / "mitapivat", y_margin=150
    )
