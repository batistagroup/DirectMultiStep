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

"""
Shape suffixes convention inspired by
https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd

B: batch size
C: the length of the input on which conditioning is done
   in our case input_max_length
L: sequence length for decoder, in our case output_max_length
M: memory length (length of sequence being attended to)
D: model dimension (sometimes called d_model or embedding_dim)
V: vocabulary size
F: feed-forward subnetwork hidden size
H: number of attention heads in a layer
K: size of each attention key or value (sometimes called d_kv)
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from typing import List, Dict, Tuple

# Define types
Tensor = torch.Tensor
BeamSearchOutput = List[List[Tuple[str, float]]]


class BeamSearch:
    def __init__(
        self,
        model: nn.Module,
        beam_size: int,
        start_idx: int,
        pad_idx: int,
        end_idx: int,
        max_length: int,
        idx_to_token: Dict[int, str],
        device: torch.device,
    ):
        self.model = model
        self.beam_size = beam_size
        self.start_idx = start_idx
        self.pad_idx = pad_idx
        self.end_idx = end_idx
        self.device = device
        self.max_length = max_length
        self.idx_to_token = idx_to_token

    def _prepare_beam_tensors(
        self, src_BC: Tensor, enc_src_BCD: Tensor
    ) -> Tuple[Tensor, Tensor, List[List[List[int]]], npt.NDArray[np.float64], Tensor]:
        B = enc_src_BCD.shape[0]
        S = self.beam_size

        beam_enc_WCD = enc_src_BCD.repeat_interleave(S, dim=0)  # W = B * S
        beam_src_WC = src_BC.repeat_interleave(S, dim=0)
        beam_src_mask_W11C = (beam_src_WC != self.pad_idx).unsqueeze(1).unsqueeze(2)
        beam_idxs_BS1_nt = [[[self.start_idx] for _ in range(S)] for _ in range(B)]
        beam_log_probs_BS_nt = np.zeros((B, S))
        cur_targets_B1 = torch.LongTensor([[self.start_idx] for _ in range(B)]).to(
            self.device
        )

        return (
            beam_enc_WCD,
            beam_src_mask_W11C,
            beam_idxs_BS1_nt,
            beam_log_probs_BS_nt,
            cur_targets_B1,
        )

    def _expand_and_normalize_candidates(
        self,
        output_BSLS: Tensor,
        beam_idxs_BSL_nt: List[List[List[int]]],
        beam_log_probs_BS_nt: npt.NDArray[np.float64],
    ) -> Tuple[List[List[float]], List[List[List[int]]]]:
        """Generate expanded candidate sequences and their probabilities."""
        B = len(beam_idxs_BSL_nt)
        S = self.beam_size

        candidate_probs_BS_nt: List[List[float]] = [[] for _ in range(B)]
        candidate_seqs_BSL_nt: List[List[List[int]]] = [
            [] for _ in range(B)
        ]  # S is actually 2S
        # candidate_probs_B_nt, candidate_seqs_B_nt =
        # _candidates(output_BSLS, beam_idxs_BS1_nt, beam_log_probs_BS_nt)

        for B_idx in range(B):  # former n
            for S_idx in range(S):  # former k
                if self.end_idx not in beam_idxs_BSL_nt[B_idx][S_idx]:
                    k_prob = beam_log_probs_BS_nt[B_idx][S_idx]
                    normalized_probs_LS = torch.log_softmax(
                        output_BSLS[B_idx, S_idx, -1, :], dim=-1
                    )
                    k_prob_vec_S = k_prob + normalized_probs_LS
                    top_k_S = torch.topk(k_prob_vec_S, S).indices

                    # find if any of the top_k_S is greater than pad_idx
                    assert torch.any(top_k_S > self.pad_idx).item() is False
                    # top_k_S[top_k_S > self.pad_idx] = self.pad_idx

                    for idx in top_k_S:
                        candidate_seqs_BSL_nt[B_idx].append(
                            beam_idxs_BSL_nt[B_idx][S_idx] + [idx.item()]
                        )
                        candidate_probs_BS_nt[B_idx].append(k_prob_vec_S[idx].item())
                else:
                    candidate_seqs_BSL_nt[B_idx].append(
                        beam_idxs_BSL_nt[B_idx][S_idx] + [self.pad_idx]
                    )
                    candidate_probs_BS_nt[B_idx].append(
                        beam_log_probs_BS_nt[B_idx][S_idx]
                    )

        return candidate_probs_BS_nt, candidate_seqs_BSL_nt

    def _select_top_k_candidates(
        self,
        candidate_probs_BS_nt: List[List[float]],
        candidate_seqs_BSL_nt: List[List[List[int]]],  # S is actually 2S
        debug: bool = False,
    ) -> List[npt.NDArray[np.int_]]:
        """Normalize probabilities and select top-k candidates."""
        B = len(candidate_probs_BS_nt)
        best_k_B_nt = []

        for B_idx in range(B):
            seq_lengths = [
                len([token for token in seq if token != self.pad_idx])
                for seq in candidate_seqs_BSL_nt[B_idx]
            ]
            normalized_probs_S = np.array(candidate_probs_BS_nt[B_idx]) / (
                np.sqrt(seq_lengths) + 1e-6
            )
            if debug:
                breakpoint()

            best_k = np.argsort(normalized_probs_S)[-self.beam_size :][::-1]
            best_k_B_nt.append(best_k)
        return best_k_B_nt

    def _generate_final_outputs(
        self, beam_idxs_BSL_nt: List[List[List[int]]], beam_log_probs_BS_nt: npt.NDArray[np.float64]
    ) -> BeamSearchOutput:
        """Convert index sequences to final outputs."""
        B = len(beam_idxs_BSL_nt)
        outputs_B2_nt: List[List[Tuple[str, float]]] = [[] for _ in range(B)]

        for B_idx in range(B):
            for S_idx in range(self.beam_size):
                output_str = ""
                for L_idx in beam_idxs_BSL_nt[B_idx][S_idx]:
                    if L_idx == self.end_idx:
                        break
                    output_str += self.idx_to_token[L_idx]
                outputs_B2_nt[B_idx].append(
                    (output_str[5:], beam_log_probs_BS_nt[B_idx][S_idx])
                )

        return outputs_B2_nt

    def decode(self, src_BC: Tensor, steps_B1: Tensor)->BeamSearchOutput:
        """
        src_BC: product + one_sm
        steps_B1: number of steps

        define S as beam_size
        define W as B*S (W for Window, a window for output)
        _nt stands for not a tensor, a regular list
        """
        self.model.eval()
        src_mask_B11C = (src_BC != self.pad_idx).unsqueeze(1).unsqueeze(2)
        enc_src_BCD = self.model.encoder(src_BC.long(), src_mask_B11C, steps_B1)
        # prepare tensors for beam search
        (
            beam_enc_WCD,
            beam_src_mask_W11C,
            beam_idxs_BS1_nt,
            beam_log_probs_BS_nt,
            cur_targets_B1,
        ) = self._prepare_beam_tensors(src_BC, enc_src_BCD)
        with torch.no_grad():
            output_BLV = self.model.decoder(
                trg_BL=cur_targets_B1,
                enc_src_BCD=enc_src_BCD,
                src_mask_B11C=src_mask_B11C,
                trg_mask_B1LL=None,
            )

        B, L, V = output_BLV.shape
        S = self.beam_size
        if self.beam_size > output_BLV.shape[-1]:
            # beam_size is greater than vocabulary, add padding
            pad_tensor = torch.full(
                (B, L, S - V), float("-inf"), device=output_BLV.device
            )
            output_BLS = torch.cat([output_BLV, pad_tensor], dim=-1)
        else:
            output_BLS = output_BLV

        normalized_probs_BLS = torch.softmax(output_BLS, dim=-1)
        # Update initial target sequences with the first step probabilities
        top_k_BS_nt = []
        for B_idx in range(B):
            sorted_idx_top_S = torch.argsort(normalized_probs_BLS[B_idx, -1, :])[
                -1 * torch.arange(1, 1 + S, 1)
            ]
            sorted_idx_top_S[sorted_idx_top_S > self.pad_idx] = self.pad_idx
            top_k_BS_nt.append(sorted_idx_top_S.tolist())
        for B_idx in range(B):
            for S_idx in range(S):
                beam_idxs_BS1_nt[B_idx][S_idx].append(
                    (chosen_idx := top_k_BS_nt[B_idx][S_idx])
                )
                beam_log_probs_BS_nt[B_idx][S_idx] += np.log(
                    normalized_probs_BLS[B_idx, -1, chosen_idx].item()
                )

        # Expand beam search over multiple decoding steps
        beam_idxs_BSL_nt = beam_idxs_BS1_nt
        for step in tqdm(range(self.max_length - 2)):
            trg_idxs_WL = (
                torch.LongTensor(beam_idxs_BSL_nt).view(B * S, -1).to(self.device)
            )
            with torch.no_grad():
                output_WLV = self.model.decoder(
                    trg_BL=trg_idxs_WL,
                    enc_src_BCD=beam_enc_WCD,
                    src_mask_B11C=beam_src_mask_W11C,
                    trg_mask_B1LL=None,
                )
                W, L, V = output_WLV.shape
                output_BSLV = output_WLV.view(B, S, L, V)
            if self.beam_size > output_WLV.shape[-1]:
                # beam_size is greater than vocabulary, add padding
                pad_tensor = torch.full(
                    (B, S, L, S - V), float("-inf"), device=output_WLV.device
                )
                output_BSLS = torch.cat([output_BSLV, pad_tensor], dim=-1)
            else:
                output_BSLS = output_BSLV
            (
                candidate_probs_BS_nt,
                candidate_seqs_BSL_nt,
            ) = self._expand_and_normalize_candidates(
                output_BSLS, beam_idxs_BS1_nt, beam_log_probs_BS_nt
            )

            best_k_B_nt = self._select_top_k_candidates(
                candidate_probs_BS_nt,
                candidate_seqs_BSL_nt,
                debug=False,  # step>113-2
            )

            for B_idx in range(B):
                for S_idx in range(S):
                    beam_idxs_BSL_nt[B_idx][S_idx] = candidate_seqs_BSL_nt[B_idx][
                        best_k_B_nt[B_idx][S_idx]
                    ]
                    beam_log_probs_BS_nt[B_idx][S_idx] = candidate_probs_BS_nt[B_idx][
                        best_k_B_nt[B_idx][S_idx]
                    ]
            if step > 150:
                break
            # if step > 113-2:
            #     breakpoint()
        return self._generate_final_outputs(beam_idxs_BSL_nt, beam_log_probs_BS_nt)
