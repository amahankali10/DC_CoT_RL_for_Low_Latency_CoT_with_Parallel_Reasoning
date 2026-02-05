import torch
from typing import List
import gc

def fill_single_example_attn_mask(full_attn_mask: torch.Tensor, segment_lengths: List[int], P: int, batch_idx: int):
    total_len = sum(segment_lengths)
    _, _, L, _ = full_attn_mask.shape
    total_len = min(total_len, L - P)
    assert total_len == sum(segment_lengths), f"total_len: {total_len}, sum(segment_lengths): {sum(segment_lengths)}"
    full_attn_mask[batch_idx, 0, P:P + total_len, P:P + total_len] = True
    block_end_idx = segment_lengths[0]
    block_num = 1

    while block_num < len(segment_lengths):
        # Worker 1 block
        block_start_idx = block_end_idx
        block_end_idx = block_start_idx + segment_lengths[block_num]
        full_attn_mask[batch_idx, 0, (P + block_end_idx):, (P + block_start_idx):(P + block_end_idx)] = False
        block_num += 1
        if block_num >= len(segment_lengths):
            break

        # Worker 2 block
        block_start_idx = block_end_idx
        block_end_idx = block_start_idx + segment_lengths[block_num]
        full_attn_mask[batch_idx, 0, (P + block_end_idx):, (P + block_start_idx):(P + block_end_idx)] = False
        block_num += 1
        if block_num >= len(segment_lengths):
            break

        # Worker 3 block
        block_start_idx = block_end_idx
        block_end_idx = block_start_idx + segment_lengths[block_num]
        full_attn_mask[batch_idx, 0, (P + block_end_idx):, (P + block_start_idx):(P + block_end_idx)] = False
        block_num += 1
        if block_num >= len(segment_lengths):
            break

        # All workers block
        block_start_idx = block_end_idx
        block_end_idx = block_start_idx + segment_lengths[block_num]
        block_num += 1
        if block_num >= len(segment_lengths):
            break

        # Single thread block
        block_start_idx = block_end_idx
        block_end_idx = block_start_idx + segment_lengths[block_num]
        block_num += 1
    
    causal_mask = torch.tril(torch.ones((total_len, total_len), dtype=torch.bool, device=full_attn_mask.device))
    full_attn_mask[batch_idx, 0, P:P + total_len, P:P + total_len] = torch.logical_and(
        causal_mask,
        full_attn_mask[batch_idx, 0, P:P + total_len, P:P + total_len]
    )
    return full_attn_mask

def get_full_parallel_attn_mask(prompt_padding_mask: torch.Tensor, segment_lengths: torch.Tensor, max_input_id_len: int):
    """
    prompt_padding_mask: boolean of shape (bs, config.prompt_len)

    segment_lengths is a torch.Tensor with the following representation:
    - is a list of lists
    - The outer list has length batch size
    - In the inner list, the first num_valid_segments
      entries are positive integers, and the rest are -1.

    max_input_id_len: int is the size of the input_ids tensor (minus the prompt length)
    """
    bs, P = prompt_padding_mask.shape
    R = max_input_id_len
    full_attn_mask = torch.zeros((bs, 1, P + R, P + R), dtype=torch.bool, device=prompt_padding_mask.device)
    segment_lengths = segment_lengths.tolist()

    for batch_idx, example_segment_lengths in enumerate(segment_lengths):
        valid_segment_lengths = []
        for length in example_segment_lengths:
            if length == -1:
                break
            valid_segment_lengths.append(length)
        
        full_attn_mask = fill_single_example_attn_mask(
            full_attn_mask=full_attn_mask,
            segment_lengths=valid_segment_lengths,
            P=P,
            batch_idx=batch_idx,
        )
    
    # Prompt to prompt: causal + padding
    full_attn_mask[:, :, :P, :P] = torch.tril((torch.ones((P, P), dtype=torch.bool, device=prompt_padding_mask.device))) # causal
    prompt_valid = prompt_padding_mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(1) # (bs, 1, 1, P)
    full_attn_mask[:, :, :P, :P] = torch.logical_and(
        prompt_valid,
        full_attn_mask[:, :, :P, :P]
    )

    # Prompt to response: zeros
    full_attn_mask[:, :, :P, P:] = False

    # Response to prompt: just padding mask
    full_attn_mask[:, :, P:, :P] = prompt_valid

    return full_attn_mask