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

from pathlib import Path
from typing import List, Tuple, cast

import lightning as L  # type:ignore
import rdkit.Chem as Chem  # type: ignore
import torch
import yaml
from rdkit import RDLogger

from DirectMultiStep.Models.Architecture import VanillaTransformerConfig
from DirectMultiStep.Models.Configure import prepare_model
from DirectMultiStep.Models.TensorGen import BeamSearchOptimized as BeamSearch
from DirectMultiStep.Utils.Dataset import RoutesDataset
from DirectMultiStep.Utils.PostProcess import find_valid_paths, process_paths
from DirectMultiStep.Utils.Visualize import draw_tree_from_path_string

RDLogger.DisableLog("rdApp.*")

data_path = Path(__file__).resolve().parent / "Data"
processed_path = data_path / "Processed"
ckpt_path = data_path / "Checkpoints"
fig_path = data_path / "Figures"

beam_size = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = "van_6x3_6x3_010"
ckpt_name = "epoch=18-step=497135.ckpt"


def canonicalize(smile: str) -> str:
    return cast(str, Chem.MolToSmiles(Chem.MolFromSmiles(smile), isomericSmiles=True))


product = canonicalize("OC(C1=C(N2N=CC=N2)C=CC(OC)=C1)=O")
sm = canonicalize("N1=CC=NN1")
n_steps = 2

L.seed_everything(42)

with open(processed_path / "character_dictionary.yaml", "rb") as file:
    data = yaml.safe_load(file)
    idx_to_token = data["invdict"]
    token_to_idx = data["smiledict"]
    product_max_length = data["product_max_length"]
    sm_max_length = data["sm_max_length"]


rds = RoutesDataset(metadata_path=str(processed_path / "character_dictionary.yaml"))
prod_tens = rds.smile_to_tokens(product, product_max_length)
sm_tens = rds.smile_to_tokens(sm, sm_max_length)
encoder_inp = torch.cat([prod_tens, sm_tens], dim=0).unsqueeze(0)
steps_tens = torch.tensor([n_steps]).unsqueeze(0)
path_start = "{'smiles':'" + product + "','children':[{'smiles':'"
path_tens = rds.path_string_to_tokens(
    path_start, max_length=None, add_eos=False
).unsqueeze(0)


van_enc_conf = VanillaTransformerConfig(
    input_dim=53,
    output_dim=53,
    input_max_length=145 + 135,
    output_max_length=1074 + 1,  # 1074 is max length
    pad_index=52,
    n_layers=6,
    ff_mult=3,
    attn_bias=False,
    ff_activation="gelu",
    hid_dim=256,
)
van_dec_conf = VanillaTransformerConfig(
    input_dim=53,
    output_dim=53,
    input_max_length=145 + 135,
    output_max_length=1074 + 1,  # 1074 is max length
    pad_index=52,
    n_layers=6,
    ff_mult=3,
    attn_bias=False,
    ff_activation="gelu",
    hid_dim=256,
)
model = prepare_model(enc_config=van_enc_conf, dec_config=van_dec_conf)

ckpt_torch = torch.load(ckpt_path / run_name / ckpt_name, map_location=device)
model.load_state_dict(ckpt_torch)
model.to(device)
model.eval()


BSObject = BeamSearch(
    model=model,
    beam_size=beam_size,
    start_idx=0,
    pad_idx=52,
    end_idx=22,
    max_length=1074,
    idx_to_token=idx_to_token,
    device=device,
)

all_beam_results_NS2: List[List[Tuple[str, float]]] = []
beam_result_BS2 = BSObject.decode(
    src_BC=encoder_inp, steps_B1=steps_tens, path_start_BL=path_tens
)
for beam_result_S2 in beam_result_BS2:
    all_beam_results_NS2.append(beam_result_S2)
print(f"{all_beam_results_NS2=}")

valid_paths_NS2n = find_valid_paths(all_beam_results_NS2, verbose=True)
correct_paths_NS2n = process_paths(
    paths_NS2n=valid_paths_NS2n,
    true_products=[product],
    true_reacs=[sm],
    commercial_stock=None,
    verbose=True,
)
# correct_paths_NS2n = valid_paths_NS2n
print(f"Length of correct paths: {len(correct_paths_NS2n[0])}")


for i, beam_result in enumerate(correct_paths_NS2n[0]):
    draw_tree_from_path_string(
        path_string=beam_result[0],
        save_path=fig_path / f"making_darid_nsteps={n_steps}_b{i + 1}",
    )
