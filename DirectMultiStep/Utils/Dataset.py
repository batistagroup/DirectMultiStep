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

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import yaml
import re


def tokenize_smile(smile: str):
    return ["<SOS>"] + list(smile) + ["?"]


def tokenize_path_string(path_string: str, add_eos: bool = True):
    pattern = re.compile(r"('smiles':|'children':|\[|\]|{|}|.)")
    tokens = ["<SOS>"] + pattern.findall(path_string)
    if add_eos:
        tokens.append("?")
    return tokens


class StepSM_Dataset_v2(Dataset):

    def __init__(
        self,
        products: List[str],
        starting_materials: List[str],
        path_strings: List[str],
        n_steps_list: List[int],
        metadata_path: str,
    ):
        self.products = products
        self.SMs = starting_materials
        self.path_strings = path_strings
        self.step_lengths = n_steps_list

        with open(metadata_path, "rb") as file:
            data = yaml.safe_load(file)
            self.token_to_idx = data["smiledict"]
            self.idx_to_token = data["invdict"]
            self.product_max_length = data["product_max_length"]
            self.sm_max_length = data["sm_max_length"]
            self.seq_out_max_length = data["seq_out_maxlength"]

        self.seq_pad_index = self.token_to_idx[" "]

    def __len__(self) -> int:
        return len(self.products)

    def char_to_ind(self, seq: List[str]) -> List[int]:
        """
        Convert a sequence of characters to token indices.
        """
        return [self.token_to_idx[char] for char in seq]

    def ind_to_seq(self, seq: List[int]) -> str:
        """
        Convert a sequence of token indices to characters.
        """
        return "".join(self.idx_to_token[index.item()] for index in seq)

    def pad(self, seq: List[int], max_length: int, pad_index: int) -> None:
        """
        Pad a sequence to the specified maximum length.

        Args:
            seq (List[int]): Sequence to pad.
            max_length (int): Maximum length of the padded sequence.
            pad_index (int): Index of the padding token.
        """
        seq.extend([pad_index] * (max_length - len(seq)))

    def smile_to_tokens(self, smile: str, max_length: int) -> torch.Tensor:
        """
        Convert a SMILES string to token indices.
        """
        tokenized = tokenize_smile(smile)
        indices = self.char_to_ind(tokenized)
        self.pad(indices, max_length, self.seq_pad_index)
        return torch.from_numpy(np.array(indices))

    def path_string_to_tokens(self, path_string: str, max_length: int) -> torch.Tensor:
        """
        Convert a path string to token indices.
        """
        tokenized = tokenize_path_string(path_string)
        indices = self.char_to_ind(tokenized)
        self.pad(indices, max_length, self.seq_pad_index)
        return torch.from_numpy(np.array(indices))

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the encoded product,
                encoded SM, and step item.
        """
        product_item = self.smile_to_tokens(
            self.products[index], self.product_max_length
        )
        one_sm_item = self.smile_to_tokens(self.SMs[index], self.sm_max_length)
        seq_encoder_item = torch.cat((product_item, one_sm_item), dim=0)
        seq_decoder_item = self.path_string_to_tokens(
            self.path_strings[index], self.seq_out_max_length
        )

        step_item = torch.tensor([self.step_lengths[index]])
        # shapes: [input_max_length], [output_max_length], int
        return seq_encoder_item, seq_decoder_item, step_item


class RoutesDataset(Dataset):

    def __init__(self):
        pass

    def __len__(self) -> int:
        return len(self.products)

    def char_to_ind(self, seq: List[str]) -> List[int]:
        """
        Convert a sequence of characters to token indices.
        """
        return [self.token_to_idx[char] for char in seq]

    def ind_to_seq(self, seq: List[int]) -> str:
        """
        Convert a sequence of token indices to characters.
        """
        return "".join(self.idx_to_token[index.item()] for index in seq)

    def pad(self, seq: List[int], max_length: int, pad_index: int) -> None:
        """
        Pad a sequence to the specified maximum length.

        Args:
            seq (List[int]): Sequence to pad.
            max_length (int): Maximum length of the padded sequence.
            pad_index (int): Index of the padding token.
        """
        seq.extend([pad_index] * (max_length - len(seq)))

    def smile_to_tokens(self, smile: str, max_length: int) -> torch.Tensor:
        """
        Convert a SMILES string to token indices.
        """
        tokenized = tokenize_smile(smile)
        indices = self.char_to_ind(tokenized)
        self.pad(indices, max_length, self.seq_pad_index)
        return torch.Tensor(indices)

    def path_string_to_tokens(self, path_string: str, max_length: Optional[int]=None, add_eos:bool=True) -> torch.Tensor:
        """
        Convert a path string to token indices.
        """
        tokenized = tokenize_path_string(path_string, add_eos=add_eos)
        indices = self.char_to_ind(tokenized)
        if max_length is not None:
            self.pad(indices, max_length, self.seq_pad_index)
        return torch.Tensor(indices)

class RoutesStepsSMDataset(RoutesDataset):

    def __init__(
        self,
        products: List[str],
        starting_materials: List[str],
        path_strings: List[str],
        n_steps_list: List[int],
        metadata_path: str,
    ):
        self.products = products
        self.SMs = starting_materials
        self.path_strings = path_strings
        self.step_lengths = n_steps_list

        with open(metadata_path, "rb") as file:
            data = yaml.safe_load(file)
            self.token_to_idx = data["smiledict"]
            self.idx_to_token = data["invdict"]
            self.product_max_length = data["product_max_length"]
            self.sm_max_length = data["sm_max_length"]
            self.seq_out_max_length = data["seq_out_maxlength"]

        self.seq_pad_index = self.token_to_idx[" "]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        product_item = self.smile_to_tokens(
            self.products[index], self.product_max_length
        )
        one_sm_item = self.smile_to_tokens(self.SMs[index], self.sm_max_length)
        seq_encoder_item = torch.cat((product_item, one_sm_item), dim=0)
        seq_decoder_item = self.path_string_to_tokens(
            self.path_strings[index], self.seq_out_max_length
        )

        step_item = torch.tensor([self.step_lengths[index]])
        return seq_encoder_item, seq_decoder_item, step_item

class RoutesStepsSMforGeneration(RoutesDataset):

    def __init__(
        self,
        products: List[str],
        starting_materials: List[str],
        path_strings: List[str],
        n_steps_list: List[int],
        metadata_path: str,
    ):
        self.products = products
        self.SMs = starting_materials
        self.path_strings = path_strings
        self.step_lengths = n_steps_list

        with open(metadata_path, "rb") as file:
            data = yaml.safe_load(file)
            self.token_to_idx = data["smiledict"]
            self.idx_to_token = data["invdict"]
            self.product_max_length = data["product_max_length"]
            self.sm_max_length = data["sm_max_length"]
            self.seq_out_max_length = data["seq_out_maxlength"]

        self.seq_pad_index = self.token_to_idx[" "]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        product_item = self.smile_to_tokens(
            self.products[index], self.product_max_length
        )
        one_sm_item = self.smile_to_tokens(self.SMs[index], self.sm_max_length)
        seq_encoder_item = torch.cat((product_item, one_sm_item), dim=0)
        seq_decoder_item = self.path_string_to_tokens(
            self.path_strings[index], self.seq_out_max_length
        )

        step_item = torch.tensor([self.step_lengths[index]])
        smile_dict = {"smiles": self.products[index], "children": [{"smiles": ""}]}
        path_start = str(smile_dict).replace(" ", "")[:-4]
        path_tens = self.path_string_to_tokens(path_start, max_length=None, add_eos=False)
        return seq_encoder_item, seq_decoder_item, step_item, path_tens

class RoutesStepsDataset(RoutesDataset):

    def __init__(
        self,
        products: List[str],
        path_strings: List[str],
        n_steps_list: List[int],
        metadata_path: str,
    ):
        self.products = products
        self.path_strings = path_strings
        self.step_lengths = n_steps_list

        with open(metadata_path, "rb") as file:
            data = yaml.safe_load(file)
            self.token_to_idx = data["smiledict"]
            self.idx_to_token = data["invdict"]
            self.product_max_length = data["product_max_length"]
            self.seq_out_max_length = data["seq_out_maxlength"]

        self.seq_pad_index = self.token_to_idx[" "]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_encoder_item = self.smile_to_tokens(
            self.products[index], self.product_max_length
        )
        seq_decoder_item = self.path_string_to_tokens(
            self.path_strings[index], self.seq_out_max_length
        )

        step_item = torch.tensor([self.step_lengths[index]])
        # shapes: [product_max_length], [output_max_length], int
        return seq_encoder_item, seq_decoder_item, step_item

class RoutesStepsforGeneration(RoutesDataset):

    def __init__(
        self,
        products: List[str],
        path_strings: List[str],
        n_steps_list: List[int],
        metadata_path: str,
    ):
        self.products = products
        self.path_strings = path_strings
        self.step_lengths = n_steps_list

        with open(metadata_path, "rb") as file:
            data = yaml.safe_load(file)
            self.token_to_idx = data["smiledict"]
            self.idx_to_token = data["invdict"]
            self.product_max_length = data["product_max_length"]
            self.seq_out_max_length = data["seq_out_maxlength"]

        self.seq_pad_index = self.token_to_idx[" "]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_encoder_item = self.smile_to_tokens(
            self.products[index], self.product_max_length
        )
        seq_decoder_item = self.path_string_to_tokens(
            self.path_strings[index], self.seq_out_max_length
        )

        step_item = torch.tensor([self.step_lengths[index]])
        # shapes: [product_max_length], [output_max_length], int
        smile_dict = {"smiles": self.products[index], "children": [{"smiles": ""}]}
        path_start = str(smile_dict).replace(" ", "")[:-4]
        # path_start = "{'smiles':'" + self.products[index] + "','children':["
        path_tens = self.path_string_to_tokens(path_start, max_length=None, add_eos=False)
        return seq_encoder_item, seq_decoder_item, step_item, path_tens
