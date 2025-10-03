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


class BatchedBeamSearch:
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
        return f"BatchedBeamSearch(beam_size={self.beam_size}, max_length={self.max_length})"

    def decode(
        self,
        src_BC: Tensor,
        steps_B1: Tensor | None,
        path_starts: list[Tensor | None] | None = None,
        target_lengths: list[int] | None = None,
        progress_bar: bool = True,
        token_processor: Callable[[list[str]], str] | None = None,
    ) -> BeamSearchOutput:
        B, C = src_BC.shape
        S = self.beam_size

        if target_lengths is None:
            target_lengths = [self.max_length] * B

        if path_starts is None:
            path_starts = [None] * B

        src_mask_B11C = (src_BC != self.pad_idx).unsqueeze(1).unsqueeze(2)
        enc_src_BCD = self.model.encoder(src_BC.long(), src_mask_B11C, steps_B1)

        beams: list[list[BeamState]] = []
        next_positions: list[int] = []
        finished: list[list[tuple[Tensor, float]]] = []
        batch_item_active: list[bool] = []

        for b in range(B):
            beam_list: list[BeamState] = []
            target_len = target_lengths[b]
            path_start = path_starts[b]

            if path_start is not None:
                start_len = path_start.size(0)
                for s in range(S):
                    seq = torch.full((target_len,), self.pad_idx, dtype=torch.long, device=self.device)
                    seq[:start_len] = path_start
                    score = 0.0 if s == 0 else float('-inf')
                    beam_list.append(BeamState(sequence=seq, score=score, active=(s == 0)))
                next_positions.append(start_len)
            else:
                for s in range(S):
                    seq = torch.full((target_len,), self.pad_idx, dtype=torch.long, device=self.device)
                    seq[0] = self.start_idx
                    score = 0.0 if s == 0 else float('-inf')
                    beam_list.append(BeamState(sequence=seq, score=score, active=(s == 0)))
                next_positions.append(1)

            beams.append(beam_list)
            finished.append([])
            batch_item_active.append(True)

        max_steps = max(target_lengths)
        pbar: Iterable[int] = tqdm(range(max_steps)) if progress_bar else range(max_steps)

        for step in pbar:
            if not any(batch_item_active):
                break

            active_batch_indices: list[int] = []
            active_beam_indices: list[int] = []
            active_sequences: list[Tensor] = []
            active_enc_outputs: list[Tensor] = []

            for b in range(B):
                if not batch_item_active[b]:
                    continue
                for s in range(S):
                    if beams[b][s].active:
                        active_batch_indices.append(b)
                        active_beam_indices.append(s)
                        seq = beams[b][s].sequence[:next_positions[b]]
                        active_sequences.append(seq)
                        active_enc_outputs.append(enc_src_BCD[b])

            if len(active_sequences) == 0:
                break

            max_seq_len = max(seq.size(0) for seq in active_sequences)
            padded_sequences = torch.full(
                (len(active_sequences), max_seq_len),
                self.pad_idx,
                dtype=torch.long,
                device=self.device
            )
            for i, seq in enumerate(active_sequences):
                padded_sequences[i, :seq.size(0)] = seq

            active_enc_stacked = torch.stack(active_enc_outputs, dim=0)
            active_src_mask = torch.ones(
                len(active_sequences), 1, 1, C,
                dtype=torch.bool,
                device=self.device
            )

            with torch.no_grad():
                output_NLV = self.model.decoder(
                    trg_BL=padded_sequences,
                    enc_src_BCD=active_enc_stacked,
                    src_mask_B11C=active_src_mask,
                    trg_mask_B1LL=None,
                )

            next_token_logits = output_NLV[:, -1, :]
            next_token_logprobs = torch.log_softmax(next_token_logits, dim=-1)

            for b in range(B):
                if not batch_item_active[b]:
                    continue

                candidates: list[dict] = []

                beam_masks = [(active_batch_indices[i] == b) for i in range(len(active_batch_indices))]
                batch_beams = [i for i, mask in enumerate(beam_masks) if mask]

                for idx in batch_beams:
                    s = active_beam_indices[idx]
                    current_score = beams[b][s].score

                    token_scores = next_token_logprobs[idx]
                    top_k = min(S, token_scores.size(0))
                    top_k_scores, top_k_tokens = torch.topk(token_scores, top_k)

                    for k in range(top_k):
                        new_seq = beams[b][s].sequence.clone()
                        new_seq[next_positions[b]] = top_k_tokens[k]
                        new_score = current_score + top_k_scores[k].item()

                        seq_len = next_positions[b] + 1
                        normalized_score = new_score / (seq_len ** 0.5)

                        candidates.append({
                            'sequence': new_seq,
                            'score': new_score,
                            'norm_score': normalized_score,
                            'finished': (top_k_tokens[k].item() == self.end_idx)
                        })

                candidates.sort(key=lambda x: x['norm_score'], reverse=True)

                finished_candidates = [c for c in candidates if c['finished']]
                continuing_candidates = [c for c in candidates if not c['finished']]

                for c in finished_candidates[:S]:
                    finished[b].append((c['sequence'][:next_positions[b]+1].clone(), c['score']))

                beam_idx = 0
                for c in continuing_candidates[:S]:
                    beams[b][beam_idx].sequence = c['sequence']
                    beams[b][beam_idx].score = c['score']
                    beams[b][beam_idx].active = True
                    beam_idx += 1

                for s in range(beam_idx, S):
                    beams[b][s].active = False

                if beam_idx == 0 or next_positions[b] >= target_lengths[b] - 1:
                    batch_item_active[b] = False
                    for s in range(S):
                        if beams[b][s].active:
                            finished[b].append(
                                (beams[b][s].sequence[:next_positions[b]].clone(), beams[b][s].score)
                            )
                            beams[b][s].active = False

            for b in range(B):
                if batch_item_active[b]:
                    next_positions[b] += 1

        results: list[list[tuple[str, float]]] = []

        for b in range(B):
            all_beams: list[tuple[Tensor, float]] = list(finished[b])

            for s in range(S):
                if beams[b][s].active:
                    all_beams.append((beams[b][s].sequence[:next_positions[b]].clone(), beams[b][s].score))

            all_beams.sort(key=lambda x: x[1], reverse=True)
            top_beams = all_beams[:S]

            batch_results: list[tuple[str, float]] = []
            for seq_tensor, score in top_beams:
                output_tokens = []
                for token_idx in seq_tensor:
                    idx = token_idx.item()
                    if idx == self.start_idx:
                        continue
                    if idx == self.end_idx:
                        break
                    output_tokens.append(self.idx_to_token[idx])

                output_str = token_processor(output_tokens) if token_processor is not None else "".join(output_tokens)
                batch_results.append((output_str, score))

            results.append(batch_results)

        return results


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
        pbar: Iterable[int] = tqdm(range(first_step, L - 1)) if progress_bar else range(first_step, L - 1)
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
