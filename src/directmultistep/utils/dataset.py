import re
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset

Tensor = torch.Tensor


def tokenize_smile(smile: str) -> list[str]:
    """Tokenizes a SMILES string by character.

    Args:
        smile: The SMILES string to tokenize.

    Returns:
        A list of tokens, including start and end of sequence tokens.
    """
    return ["<SOS>"] + list(smile) + ["?"]


def tokenize_smile_atom(smile: str, has_atom_types: list[str], mask: bool = False) -> list[str]:
    """Tokenizes a SMILES string, considering atom types of up to two characters.

    Args:
        smile: The SMILES string to tokenize.
        has_atom_types: A list of atom types to consider (e.g., ["Cl", "Br"]).
        mask: If True, replaces all atom tokens with "J".

    Returns:
        A list of tokens, including start and end of sequence tokens.
    """
    tokens = []
    i = 0
    while i < len(smile):
        if i < len(smile) - 1 and smile[i : i + 2] in has_atom_types:
            tokens.append("J" if mask else smile[i : i + 2])
            i += 2
        else:
            tokens.append("J" if mask else smile[i])
            i += 1
    return ["<SOS>"] + tokens + ["?"]


def tokenize_context(context_list: list[str]) -> list[str]:
    """Tokenizes a list of context strings.

    Args:
        context_list: A list of context strings to tokenize.

    Returns:
        A list of tokens, including context start, separator, and end tokens.
    """
    tokens = ["<context>"]
    for context in context_list:
        tokens.extend(tokenize_path_string(context, add_sos=False, add_eos=False))
        tokens.append("<sep>")
    tokens.append("</context>")
    return tokens


def tokenize_path_string(path_string: str, add_sos: bool = True, add_eos: bool = True) -> list[str]:
    """Tokenizes a path string based on a regular expression.

    Args:
        path_string: The path string to tokenize.
        add_sos: If True, adds a start of sequence token.
        add_eos: If True, adds an end of sequence token.

    Returns:
        A list of tokens.
    """
    pattern = re.compile(r"('smiles':|'children':|\[|\]|{|}|.)")
    tokens = ["<SOS>"] if add_sos else []
    tokens.extend(pattern.findall(path_string))
    if add_eos:
        tokens.append("?")
    return tokens


class RoutesProcessing(Dataset[tuple[Tensor, ...]]):
    """Base class for processing route data."""

    def __init__(
        self,
        metadata_path: Path,
    ) -> None:
        """Initializes the RoutesProcessing class.

        Args:
            metadata_path: Path to the metadata file (YAML).
        """
        with open(metadata_path, "rb") as file:
            data = yaml.safe_load(file)
            self.token_to_idx = data["smiledict"]
            self.idx_to_token = data["invdict"]
            self.product_max_length = data["product_max_length"]
            self.seq_out_max_length = data["seq_out_maxlength"]
            self.sm_max_length = data["sm_max_length"]

        self.seq_pad_index = self.token_to_idx[" "]

    def char_to_ind(self, seq: list[str]) -> list[int]:
        """Converts a sequence of characters to token indices.

        Args:
            seq: A list of characters.

        Returns:
            A list of token indices.
        """
        return [self.token_to_idx[char] for char in seq]

    def pad(self, seq: list[int], max_length: int, pad_index: int) -> None:
        """Pads a sequence to the specified maximum length.

        Args:
            seq: Sequence to pad.
            max_length: Maximum length of the padded sequence.
            pad_index: Index of the padding token.
        """
        seq.extend([pad_index] * (max_length - len(seq)))

    def smile_to_tokens(self, smile: str, max_length: int) -> Tensor:
        """Converts a SMILES string to token indices.

        Args:
            smile: The SMILES string to convert.
            max_length: The maximum length of the tokenized sequence.

        Returns:
            A tensor of token indices.
        """
        tokenized = tokenize_smile(smile)
        indices = self.char_to_ind(tokenized)
        self.pad(indices, max_length, self.seq_pad_index)
        return Tensor(indices)

    def path_string_to_tokens(self, path_string: str, max_length: int | None = None, add_eos: bool = True) -> Tensor:
        """Converts a path string to token indices.

        Args:
            path_string: The path string to convert.
            max_length: The maximum length of the tokenized sequence. If None, no padding is applied.
            add_eos: If True, adds an end of sequence token.

        Returns:
            A tensor of token indices.
        """
        tokenized = tokenize_path_string(path_string, add_eos=add_eos)
        indices = self.char_to_ind(tokenized)
        if max_length is not None:
            self.pad(indices, max_length, self.seq_pad_index)
        return Tensor(indices)


class RoutesDataset(RoutesProcessing):
    """Dataset for multi-step reaction routes."""

    def __init__(
        self,
        metadata_path: Path,
        products: list[str],
        path_strings: list[str],
        n_steps_list: list[int],
        starting_materials: list[str] | None = None,
        mode: str = "training",
        name_idx: dict[str, list[int]] | None = None,
    ) -> None:
        """Initializes the RoutesDataset.

        Args:
            metadata_path: Path to the metadata file (YAML).
            products: List of product SMILES strings.
            path_strings: List of reaction path strings.
            n_steps_list: List of integers representing the number of steps in each path.
            starting_materials: List of starting material SMILES strings.
            mode: Either "training" or "generation".
            name_idx: A dictionary mapping names to lists of indices.
        """
        super().__init__(metadata_path)
        self.products = products
        self.path_strings = path_strings
        self.step_lengths = n_steps_list
        self.sms = starting_materials
        # name_idx is an optional attribute that shows labels for items in the dataset
        # currently used for evals on pharma compounds
        self.name_idx = name_idx
        assert mode in ["training", "generation"], "mode must be either 'training' or 'generation'"
        self.mode = mode

    def __repr__(self) -> str:
        """Returns a string representation of the dataset."""
        sms_str = "SM (enabled)" if self.sms is not None else "SM (disabled)"
        return f"RoutesDataset(mode={self.mode}, len={len(self)}, {sms_str})"

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        """Retrieves an item from the dataset.

        Args:
            index: The index of the item to retrieve.

        Returns:
            A tuple of tensors representing the input and output data.
        """
        if self.mode == "training":
            if self.sms is not None:
                return self.get_training_with_sm(index)
            else:
                return self.get_training_no_sm(index)
        elif self.mode == "generation":
            if self.sms is not None:
                return self.get_generation_with_sm(index)
            else:
                return self.get_generation_no_sm(index)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def __len__(self) -> int:
        """Returns the number of items in the dataset."""
        return len(self.products)

    def get_training_with_sm(self, index: int) -> tuple[Tensor, ...]:
        """Retrieves a training item with starting materials.

        Args:
            index: The index of the item to retrieve.

        Returns:
            A tuple of tensors: encoder input, decoder input, and step length.
        """
        assert self.sms is not None, "starting materials are not provided"
        product_item = self.smile_to_tokens(self.products[index], self.product_max_length)
        one_sm_item = self.smile_to_tokens(self.sms[index], self.sm_max_length)
        seq_encoder_item = torch.cat((product_item, one_sm_item), dim=0)
        seq_decoder_item = self.path_string_to_tokens(self.path_strings[index], self.seq_out_max_length)

        step_item = Tensor([self.step_lengths[index]])
        return seq_encoder_item, seq_decoder_item, step_item

    def get_generation_with_sm(self, index: int) -> tuple[Tensor, ...]:
        """Retrieves a generation item with starting materials.

        Args:
            index: The index of the item to retrieve.

        Returns:
            A tuple of tensors: encoder input, step length, and initial path tensor.
        """
        assert self.sms is not None, "starting materials are not provided"
        product_item = self.smile_to_tokens(self.products[index], self.product_max_length)
        one_sm_item = self.smile_to_tokens(self.sms[index], self.sm_max_length)
        seq_encoder_item = torch.cat((product_item, one_sm_item), dim=0)

        step_item = Tensor([self.step_lengths[index]])
        smile_dict = {"smiles": self.products[index], "children": [{"smiles": ""}]}
        path_start = str(smile_dict).replace(" ", "")[:-4]
        path_tens = self.path_string_to_tokens(path_start, max_length=None, add_eos=False)
        return seq_encoder_item, step_item, path_tens

    def get_training_no_sm(self, index: int) -> tuple[Tensor, ...]:
        """Retrieves a training item without starting materials.

        Args:
            index: The index of the item to retrieve.

        Returns:
            A tuple of tensors: encoder input, decoder input, and step length.
        """
        seq_encoder_item = self.smile_to_tokens(self.products[index], self.product_max_length)
        seq_decoder_item = self.path_string_to_tokens(self.path_strings[index], self.seq_out_max_length)

        step_item = Tensor([self.step_lengths[index]])
        # shapes: [product_max_length], [output_max_length], int
        return seq_encoder_item, seq_decoder_item, step_item

    def get_generation_no_sm(self, index: int) -> tuple[Tensor, ...]:
        """Retrieves a generation item without starting materials.

        Args:
            index: The index of the item to retrieve.

        Returns:
            A tuple of tensors: encoder input, step length, and initial path tensor.
        """
        seq_encoder_item = self.smile_to_tokens(self.products[index], self.product_max_length)

        step_item = Tensor([self.step_lengths[index]])
        # shapes: [product_max_length], [output_max_length], int
        smile_dict = {"smiles": self.products[index], "children": [{"smiles": ""}]}
        path_start = str(smile_dict).replace(" ", "")[:-4]
        # path_start = "{'smiles':'" + self.products[index] + "','children':["
        path_tens = self.path_string_to_tokens(path_start, max_length=None, add_eos=False)
        return seq_encoder_item, step_item, path_tens
