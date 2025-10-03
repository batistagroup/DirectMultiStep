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

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import tqdm

from directmultistep.utils.logging_config import logger

Tensor = torch.Tensor
BeamSearchOutput = list[list[tuple[str, float]]]


@dataclass
class BeamState:
    sequence: Tensor
    score: float
    active: bool


class VectorizedBatchedBeamSearch:
    def __init__(
        self,
        model: nn.Module,
        beam_size: int,
        start_idx: int,
        pad_idx: int,
        end_idx: int,
        max_length: int,
        idx_to_token: dict[int, str],
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

        # Pre-allocate reusable tensors
        self._init_reusable_tensors()

    def __repr__(self) -> str:
        return f"VectorizedBatchedBeamSearch(beam_size={self.beam_size}, max_length={self.max_length})"

    def _init_reusable_tensors(self):
        """Pre-allocate tensors that can be reused across decode calls"""
        S = self.beam_size
        # Pre-compute beam indices for gather operations
        self.beam_indices = torch.arange(S, device=self.device)
        self.beam_offsets = torch.arange(S, device=self.device).unsqueeze(-1) * S

    def decode(
        self,
        src_BC: Tensor,
        steps_B1: Tensor | None,
        path_starts: list[Tensor | None] | None = None,
        target_lengths: list[int] | None = None,
        progress_bar: bool = True,
        token_processor: Callable[[list[str]], str] | None = None,
    ) -> BeamSearchOutput:
        """
        Optimized fully vectorized beam search.
        """
        B, C = src_BC.shape
        S = self.beam_size
        L = self.max_length
        V = len(self.idx_to_token)

        # Detect the dtype from input (or could check model.parameters())
        dtype = src_BC.dtype

        # Encode source sequences once
        src_mask_B11C = (src_BC != self.pad_idx).unsqueeze(1).unsqueeze(2)
        enc_src_BCD = self.model.encoder(src_BC.long(), src_mask_B11C, steps_B1)

        # Optimization 1: Use view instead of reshape for encoder expansion
        enc_src_BSCD = enc_src_BCD.unsqueeze(1).expand(B, S, -1, -1).contiguous().view(B * S, -1, enc_src_BCD.size(-1))
        src_mask_BS11C = src_mask_B11C.unsqueeze(1).expand(B, S, -1, -1, -1).contiguous().view(B * S, 1, 1, C)

        # Initialize beam tensors
        sequences_BSL = torch.full((B, S, L), self.pad_idx, dtype=torch.long, device=self.device)
        scores_BS = torch.full((B, S), float("-inf"), dtype=dtype, device=self.device)
        scores_BS[:, 0] = 0.0
        active_BS = torch.ones((B, S), dtype=torch.bool, device=self.device)

        # Handle path starts
        first_steps_B = torch.ones(B, dtype=torch.long, device=self.device)
        if path_starts:
            for b, path in enumerate(path_starts):
                if path is not None:
                    path_len = path.size(0)
                    sequences_BSL[b, :, :path_len] = path.unsqueeze(0).expand(S, -1)
                    first_steps_B[b] = path_len
                else:
                    sequences_BSL[b, :, 0] = self.start_idx
        else:
            sequences_BSL[:, :, 0] = self.start_idx

        first_step = first_steps_B.min().item()
        max_steps = L - 1 if target_lengths is None else max(target_lengths)

        # Pre-allocate reusable tensors for the loop
        # batch_indices = torch.arange(B, device=self.device).unsqueeze(1).expand(B, S)
        # beam_expand_indices = self.beam_indices.unsqueeze(0).expand(B, -1)

        # Pre-allocate candidate selection tensors
        candidate_buffer = torch.zeros((B, S * S), device=self.device, dtype=dtype)
        beam_idx_buffer = torch.zeros((B, S * S), dtype=torch.long, device=self.device)
        token_idx_buffer = torch.zeros((B, S * S), dtype=torch.long, device=self.device)

        # Pre-compute beam index patterns for candidate expansion
        beam_idx_pattern = self.beam_indices.unsqueeze(1).expand(-1, S).reshape(-1)

        pbar: Iterable[int] = (
            tqdm(range(first_step, max_steps), desc="Beam search", dynamic_ncols=True)
            if progress_bar
            else range(first_step, max_steps)
        )
        # OPTIMIZATION: Pre-compute beam patterns once (outside loop ideally)
        beam_idx_pattern = torch.arange(S, device=self.device).unsqueeze(1).expand(S, S).reshape(-1)


        for step in pbar:
            # Check which batches have started decoding
            step_active_B = step >= first_steps_B

            # Check for end tokens
            has_end_token_BS = (sequences_BSL[:, :, :step] == self.end_idx).any(dim=2)
            active_BS &= ~has_end_token_BS

            # Check if all beams for each batch are inactive
            batch_finished_B = ~active_BS.any(dim=1) | ~step_active_B

            if batch_finished_B.all():
                break

            if progress_bar and batch_finished_B.any():
                finished_count = batch_finished_B.sum().item()
                pbar.set_postfix({"Finished batches": f"{finished_count}/{B}"})

            # Create mask for active beams
            active_mask_BS = active_BS & step_active_B.unsqueeze(1)

            if not active_mask_BS.any():
                continue

            # KEEP ORIGINAL: Only process active sequences
            active_mask_flat = active_mask_BS.view(B * S)
            n_active = active_mask_flat.sum().item()

            if n_active == 0:
                continue

            # Get active sequence indices
            active_indices = active_mask_flat.nonzero(as_tuple=True)[0]

            # Process only active sequences
            sequences_flat = sequences_BSL.view(B * S, L)
            active_sequences = sequences_flat[active_indices, :step]
            active_enc_src = enc_src_BSCD[active_indices]
            active_src_mask = src_mask_BS11C[active_indices]

            # Forward pass only for active beams
            with torch.no_grad():
                active_output = self.model.decoder(
                    trg_BL=active_sequences,
                    enc_src_BCD=active_enc_src,
                    src_mask_B11C=active_src_mask,
                    trg_mask_B1LL=None,
                )

            # Get log probabilities
            active_log_probs = torch.log_softmax(active_output[:, -1, :], dim=-1)

            # Scatter back to full tensor
            log_probs_BSV = torch.full((B * S, active_log_probs.size(-1)), float("-inf"), dtype=dtype, device=self.device)
            log_probs_BSV[active_indices] = active_log_probs
            log_probs_BSV = log_probs_BSV.view(B, S, -1)

            # Create expansion mask for first step logic
            is_first_step_B = (step == first_steps_B)
            expand_mask_BS = torch.ones((B, S), dtype=torch.bool, device=self.device)
            expand_mask_BS[is_first_step_B, 1:] = False

            # Combined mask for beams that should generate candidates
            generate_mask_BS = active_mask_BS & expand_mask_BS

            # Get top-k tokens for each beam
            top_k_scores_BSS, top_k_tokens_BSS = torch.topk(log_probs_BSV, S, dim=-1)

            # Calculate candidate scores
            candidate_scores_BSS = scores_BS.unsqueeze(-1) + top_k_scores_BSS

            # For first step beams, only use first beam's candidates
            if is_first_step_B.any():
                first_step_mask = is_first_step_B.view(-1, 1, 1)
                first_beam_scores = candidate_scores_BSS[:, 0:1, :].expand(B, S, S)
                candidate_scores_BSS = torch.where(first_step_mask, first_beam_scores, candidate_scores_BSS)

            # Mask out invalid candidates
            generate_mask_BSS = generate_mask_BS.unsqueeze(-1)
            candidate_scores_BSS = candidate_scores_BSS.masked_fill(~generate_mask_BSS, float("-inf"))

            # Flatten candidates
            candidate_buffer = candidate_scores_BSS.view(B, S * S)
            beam_idx_buffer = beam_idx_pattern.unsqueeze(0).expand(B, -1)
            token_idx_buffer = top_k_tokens_BSS.view(B, S * S)

            # Handle inactive beams
            inactive_mask_BS = ~active_BS & (scores_BS > float("-inf"))

            # OPTIMIZATION: Use pre-allocated tensors and efficient concatenation
            n_candidates = S * S
            all_scores = torch.cat([candidate_buffer,
                                   torch.where(inactive_mask_BS, scores_BS, torch.tensor(float("-inf"), device=self.device))], dim=1)

            all_beam_indices = torch.cat([beam_idx_buffer,
                                           torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)], dim=1)

            all_token_indices = torch.cat([token_idx_buffer,
                                            torch.full((B, S), -1, dtype=torch.long, device=self.device)], dim=1)

            # Compute sequence lengths for normalization
            seq_lengths_BS = (sequences_BSL != self.pad_idx).sum(dim=-1).float()

            # OPTIMIZATION: More efficient length computation
            all_seq_lengths = torch.cat([
                seq_lengths_BS.unsqueeze(-1).expand(B, S, S).reshape(B, n_candidates) + 1,
                seq_lengths_BS
            ], dim=1)

            # Normalize
            normalized_scores = all_scores / (all_seq_lengths.sqrt() + 1e-6)

            # Select top S beams for each batch
            _, top_indices = torch.topk(normalized_scores, S, dim=1)

            # Gather selected beams
            selected_beam_indices = torch.gather(all_beam_indices, 1, top_indices)
            selected_token_indices = torch.gather(all_token_indices, 1, top_indices)
            selected_scores = torch.gather(all_scores, 1, top_indices)

            # Update sequences
            gather_indices = selected_beam_indices.unsqueeze(-1).expand(B, S, L)
            sequences_BSL = torch.gather(sequences_BSL, 1, gather_indices)

            # Add new tokens where applicable
            mask_add_token = selected_token_indices != -1
            batch_coords, beam_coords = torch.where(mask_add_token)
            if batch_coords.numel() > 0:
                sequences_BSL[batch_coords, beam_coords, step] = selected_token_indices[batch_coords, beam_coords]

            # Update active status
            has_end_in_selected = (selected_token_indices == self.end_idx) | \
                                 (sequences_BSL[:, :, :step] == self.end_idx).any(dim=2)
            was_inactive = selected_token_indices == -1

            # Update all states
            scores_BS = selected_scores
            active_BS = ~has_end_in_selected & ~was_inactive

        # Extract final results (unchanged)
        outputs_BS2_nt: list[list[tuple[str, float]]] = []

        for b in range(B):
            batch_results = []
            for s in range(S):
                if scores_BS[b, s] == float("-inf"):
                    continue

                output_tokens = []
                for idx in sequences_BSL[b, s]:
                    if idx == self.pad_idx:
                        break
                    if idx == self.start_idx:
                        continue
                    if idx == self.end_idx:
                        break
                    output_tokens.append(self.idx_to_token[idx.item()])

                output_str = token_processor(output_tokens) if token_processor else "".join(output_tokens)
                batch_results.append((output_str, scores_BS[b, s].item()))

            outputs_BS2_nt.append(batch_results)

        return outputs_BS2_nt


class BeamSearchOptimized:
    def __init__(
        self,
        model: nn.Module,
        beam_size: int,
        start_idx: int,
        pad_idx: int,
        end_idx: int,
        max_length: int,
        idx_to_token: dict[int, str],
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

    def __repr__(self) -> str:
        return f"BeamSearchOptimized(beam_width={self.beam_size}, max_length={self.max_length})"

    def decode(
        self,
        src_BC: Tensor,
        steps_B1: Tensor | None,
        path_start_BL: Tensor | None = None,
        progress_bar: bool = True,
        token_processor: Callable[[list[str]], str] | None = None,
    ) -> BeamSearchOutput:
        """
        src_BC: product + one_sm (B, C)
        steps_B1: number of steps (B, 1)

        Define S as beam_size.
        Define W as B*S (W for Window, a window for output).
        _nt stands for not a tensor, a regular list.
        """
        B, C = src_BC.shape
        S = self.beam_size
        L = self.max_length

        src_mask_B11C = (src_BC != self.pad_idx).unsqueeze(1).unsqueeze(2)
        enc_src_BCD = self.model.encoder(src_BC.long(), src_mask_B11C, steps_B1)
        beam_enc_WCD = enc_src_BCD.repeat_interleave(S, dim=0)

        beam_src_WC = src_BC.repeat_interleave(S, dim=0)
        beam_src_mask_W11C = (beam_src_WC != self.pad_idx).unsqueeze(1).unsqueeze(2)

        beam_idxs_WL = torch.full((B * S, L), self.pad_idx, dtype=torch.long, device=self.device)
        if path_start_BL is None:
            beam_idxs_WL[:, 0] = self.start_idx
            first_step = 1
            beam_log_probs_W = torch.zeros(B * S, device=self.device)
        else:
            beam_idxs_WL[:, : path_start_BL.size(1)] = path_start_BL
            first_step = path_start_BL.size(1)
            beam_log_probs_W = torch.zeros(B * S, device=self.device)

        finished_sequences_W = torch.zeros(B * S, dtype=torch.bool, device=self.device)
        logger.info(
            f"Generating routes with beam size {S}. The progress bar may end early if all beams find end token."
        )
        pbar: Iterable[int] = (
            tqdm(range(first_step, L - 1), dynamic_ncols=True) if progress_bar else range(first_step, L - 1)
        )
        for step in pbar:
            with torch.no_grad():
                output_WLV = self.model.decoder(
                    trg_BL=beam_idxs_WL[:, :step],
                    enc_src_BCD=beam_enc_WCD,
                    src_mask_B11C=beam_src_mask_W11C,
                    trg_mask_B1LL=None,
                )
            W, _, V = output_WLV.shape
            output_WV = output_WLV[:, -1, :]
            log_probs_WV = torch.log_softmax(output_WV, dim=-1)

            finished_sequences_W = torch.any(beam_idxs_WL == self.end_idx, dim=-1)
            active_mask_W = ~finished_sequences_W
            if finished_sequences_W.all():
                break

            if step == first_step:
                log_probs_BSV = log_probs_WV.view(B, S, -1)
                log_probs_WS, top_k_idxs_WS = torch.topk(log_probs_BSV[:, 0, :], S, dim=-1)
                beam_log_probs_W = log_probs_WS.view(B * S)
                beam_idxs_WL[:, step] = top_k_idxs_WS.view(B * S)
            else:
                active_WV = active_mask_W.unsqueeze(1).expand(-1, V)
                cur_log_probs_WV = beam_log_probs_W.unsqueeze(1) + log_probs_WV

                _, act_top_k_idxs_WS = torch.topk(cur_log_probs_WV[active_WV].view(-1, V), S, dim=-1)
                act_top_k_idxs_BSS = act_top_k_idxs_WS.view(B, -1, S)

                active_WL = active_mask_W.unsqueeze(-1).repeat(1, L)
                active_beams_WL = beam_idxs_WL[active_WL].view(-1, L)
                active_beams_BSL = active_beams_WL.view(B, -1, L)
                _S = active_beams_BSL.size(1)
                active_beams_BSSL = active_beams_BSL.unsqueeze(2).repeat(1, 1, S, 1)
                active_beams_BSSL[..., step] = act_top_k_idxs_BSS
                active_beams_BSsqL = active_beams_BSSL.view(B, -1, L)
                cur_log_probs_WS = cur_log_probs_WV[active_mask_W].view(-1, V).gather(1, act_top_k_idxs_WS)

                sequence_lengths_WL = (active_beams_WL.ne(self.pad_idx).sum(dim=1).float()).unsqueeze(1)

                normalized_act_log_probs_WS = cur_log_probs_WS / (sequence_lengths_WL.sqrt() + 1e-6)
                normalized_act_log_probs_BSsq = normalized_act_log_probs_WS.view(B, -1)
                _, best_idxs_BS = normalized_act_log_probs_BSsq.topk(S, dim=-1)

                active_beams_WL = active_beams_BSsqL.view(-1, L).gather(
                    0, best_idxs_BS.view(-1).unsqueeze(-1).expand(-1, L)
                )
                active_log_probs_W = cur_log_probs_WS.view(-1).gather(0, best_idxs_BS.view(-1))

                active_beams_BSL = active_beams_WL.view(B, -1, L)
                active_log_probs_BS = active_log_probs_W.view(B, -1)

                inactive_beams_WL = beam_idxs_WL[~active_WL]
                inactive_log_probs_W = beam_log_probs_W[~active_mask_W]
                inactive_beams_BSL = inactive_beams_WL.view(B, -1, L)
                inactive_log_probs_BS = inactive_log_probs_W.view(B, -1)

                both_beams_BSL = torch.cat([active_beams_BSL, inactive_beams_BSL], dim=1)
                both_log_probs_BS = torch.cat([active_log_probs_BS, inactive_log_probs_BS], dim=1)

                both_beams_WL = both_beams_BSL.view(-1, L)
                both_log_probs_W = both_log_probs_BS.view(-1)

                both_seq_lengths_W = both_beams_WL.ne(self.pad_idx).sum(dim=1).float()
                both_normalized_log_probs_WS = both_log_probs_W / (both_seq_lengths_W.sqrt() + 1e-6)
                both_normalized_log_probs_BSsq = both_normalized_log_probs_WS.view(B, -1)
                _, best_idxs_BS = both_normalized_log_probs_BSsq.topk(S, dim=-1)

                beam_idxs_WL = both_beams_BSL.gather(1, best_idxs_BS.unsqueeze(-1).expand(-1, -1, L)).view(-1, L)
                beam_log_probs_W = both_log_probs_BS.gather(1, best_idxs_BS).view(-1)

        beam_idxs_BSL = beam_idxs_WL.view(B, S, L)
        beam_log_probs_BS = beam_log_probs_W.view(B, S)

        outputs_BS2_nt: list[list[tuple[str, float]]] = [[] for _ in range(B)]

        for b in range(B):
            for s in range(S):
                output_tokens = []
                for L_idx in beam_idxs_BSL[b, s]:
                    if L_idx == self.start_idx:
                        continue
                    if L_idx == self.end_idx:
                        break
                    output_tokens.append(self.idx_to_token[L_idx.item()])

                output_str = token_processor(output_tokens) if token_processor is not None else "".join(output_tokens)

                log_prob = beam_log_probs_BS[b, s].item()
                outputs_BS2_nt[b].append((output_str, log_prob))

        return outputs_BS2_nt
