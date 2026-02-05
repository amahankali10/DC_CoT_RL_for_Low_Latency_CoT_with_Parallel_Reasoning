import copy
from transformers import AutoTokenizer
import torch

THINK_START = "<think>"
SPAWN_WORKERS = "<spawn_workers>"
UNSPAWN_WORKERS = "</spawn_workers>"
WORKER_STARTS = ["<worker_1>", "<worker_2>", "<worker_3>"]
WORKER_ENDS = ["</worker_1>", "</worker_2>", "</worker_3>"]

IGNORE_INDEX = -100

"""
Returns
- input_ids_parts
- position_ids_parts

Breaks up the response into parts (single-thread generation, worker generation)
This is then used to create the attention mask for training parallel generation.
"""
def parse_response(response, tokenizer):
    input_ids_parts = []
    position_ids_parts = []
    labels_parts = []

    # First part - from <think> to <spawn_workers>
    think_start = response.find(THINK_START)
    spawn_workers_inc = response.find(SPAWN_WORKERS) + len(SPAWN_WORKERS)
    first_st_part = response[think_start:spawn_workers_inc]
    first_st_part = tokenizer.encode(first_st_part, add_special_tokens=False)
    input_ids_parts.append(first_st_part)
    position_ids_parts.append([i for i in range(len(first_st_part))])
    first_label_part = first_st_part[1:] + [IGNORE_INDEX]
    labels_parts.append(first_label_part)
    response = response[spawn_workers_inc:]

    while True:
        # Find <worker_1> to </worker_1>, etc.
        worker_parts = []
        worker_parts_position_ids = []
        worker_parts_labels = []
        for worker_start, worker_end in zip(WORKER_STARTS, WORKER_ENDS):
            worker_start_pos = response.find(worker_start)
            worker_start_pos_inc = worker_start_pos + len(worker_start)
            worker_end_pos_inc = response.find(worker_end) + len(worker_end)
            current_worker_part = tokenizer.encode(worker_start, add_special_tokens=False)
            current_worker_label = [IGNORE_INDEX] * (len(current_worker_part))
            current_worker_part.extend(tokenizer.encode(response[worker_start_pos_inc:worker_end_pos_inc], add_special_tokens=False))
            current_worker_label.extend(tokenizer.encode(response[worker_start_pos_inc:worker_end_pos_inc], add_special_tokens=False))
            current_worker_label = current_worker_label[1:] + [IGNORE_INDEX]

            worker_parts.append(current_worker_part)
            worker_parts_labels.append(current_worker_label)
            response = response[worker_end_pos_inc:]

            prev_pos_id = position_ids_parts[-1][-1]
            current_worker_part_position_ids = [prev_pos_id + 1 + i for i in range(len(current_worker_part))]
            worker_parts_position_ids.append(current_worker_part_position_ids)
        
        # Append worker parts twice, once with them being independent, and once with them being causal
        # For the causal part, there are no labels
        prev_st_pos_id = position_ids_parts[-1][-1]
        all_workers = []
        for part in worker_parts:
            input_ids_parts.append(copy.deepcopy(part))
            all_workers.extend(part)
        for pos_ids in worker_parts_position_ids:
            position_ids_parts.append(copy.deepcopy(pos_ids))
        for w_labels in worker_parts_labels:
            labels_parts.append(copy.deepcopy(w_labels))
        
        all_workers_pos_ids = [prev_st_pos_id + 1 + i for i in range(len(all_workers))]
        all_workers_labels = [IGNORE_INDEX] * (len(all_workers))
        input_ids_parts.append(all_workers)
        position_ids_parts.append(all_workers_pos_ids)
        labels_parts.append(all_workers_labels)

        # Next part, from </spawn_workers> to <spawn_workers> (or end of response)
        unspawn_workers = response.find(UNSPAWN_WORKERS)
        unspawn_workers_inc = unspawn_workers + len(UNSPAWN_WORKERS)
        next_spawn_workers = response.find(SPAWN_WORKERS)
        if next_spawn_workers == -1:
            next_st_part_text = response[unspawn_workers_inc:]
        else:
            next_spawn_workers_inc = next_spawn_workers + len(SPAWN_WORKERS)
            next_st_part_text = response[unspawn_workers_inc:next_spawn_workers_inc]
        next_st_part = tokenizer.encode(UNSPAWN_WORKERS, add_special_tokens=False)
        next_st_part_labels = [IGNORE_INDEX] * len(next_st_part)
        next_st_part.extend(tokenizer.encode(next_st_part_text, add_special_tokens=False))
        next_st_part_labels.extend(tokenizer.encode(next_st_part_text, add_special_tokens=False))
        if next_spawn_workers == -1:
            next_st_part.append(tokenizer.eos_token_id)
            next_st_part_labels.append(tokenizer.eos_token_id)
        next_st_part_labels = next_st_part_labels[1:] + [IGNORE_INDEX]

        input_ids_parts.append(next_st_part)
        prev_pos_id = position_ids_parts[-1][-1]
        next_st_part_position_ids = [prev_pos_id + 1 + i for i in range(len(next_st_part))]
        position_ids_parts.append(next_st_part_position_ids)
        labels_parts.append(next_st_part_labels)

        if next_spawn_workers == -1:
            break

    return input_ids_parts, position_ids_parts, labels_parts

def get_segment_lengths(input_ids_parts):
    length_list = []
    for segment in input_ids_parts:
        length_list.append(len(segment))
    return length_list

"""
This method is similar to create_attention_mask,
but we directly give it the list of segment lengths
instead of the input_ids_parts. This also assumes that
the prompt is already part of the first segment of input_ids_parts.
"""
def create_attention_mask_from_segment_lengths(segment_length_list):
    total_len = sum(segment_length_list)
    attn_mask = torch.ones((total_len, total_len), dtype=torch.bool)
    block_end_idx = segment_length_list[0]
    block_num = 1

    while block_num < len(segment_length_list):
        # Worker 1 block
        block_start_idx = block_end_idx
        block_end_idx = block_start_idx + segment_length_list[block_num]
        attn_mask[block_end_idx:, block_start_idx:block_end_idx] = False
        block_num += 1

        # Worker 2 block
        block_start_idx = block_end_idx
        block_end_idx = block_start_idx + segment_length_list[block_num]
        attn_mask[block_end_idx:, block_start_idx:block_end_idx] = False
        block_num += 1

        # Worker 3 block
        block_start_idx = block_end_idx
        block_end_idx = block_start_idx + segment_length_list[block_num]
        attn_mask[block_end_idx:, block_start_idx:block_end_idx] = False
        block_num += 1

        # All workers block
        block_start_idx = block_end_idx
        block_end_idx = block_start_idx + segment_length_list[block_num]
        block_num += 1

        # Single thread block
        block_start_idx = block_end_idx
        block_end_idx = block_start_idx + segment_length_list[block_num]
        block_num += 1
    
    causal_mask = torch.tril(torch.ones((total_len, total_len), dtype=torch.bool))
    attn_mask = torch.logical_and(attn_mask, causal_mask)
    return attn_mask

def process_completion(response, tokenizer, prompt_token_ids):

    input_ids_parts, position_ids_parts, labels_parts = parse_response(response, tokenizer)
    assert len(input_ids_parts) == len(position_ids_parts)
    assert len(input_ids_parts) == len(labels_parts)
    for i in range(len(input_ids_parts)):
        assert len(input_ids_parts[i]) == len(position_ids_parts[i])
        assert len(input_ids_parts[i]) == len(labels_parts[i])
    
    full_input_ids = copy.deepcopy(prompt_token_ids)
    for block in input_ids_parts:
        full_input_ids.extend(block)
    
    full_labels = [IGNORE_INDEX] * (len(prompt_token_ids) - 1)
    full_labels.append(input_ids_parts[0][0])
    for block in labels_parts:
        full_labels.extend(block)
    
    position_id_offset = len(prompt_token_ids)
    full_position_ids = []
    for i, _ in enumerate(prompt_token_ids):
        full_position_ids.append(i)
    for block in position_ids_parts:
        offset_block = [pos_id + position_id_offset for pos_id in block]
        full_position_ids.extend(offset_block)
    
    segment_lengths = get_segment_lengths(input_ids_parts)
    segment_lengths[0] += len(prompt_token_ids)
    
    return {
        "input_ids": full_input_ids,
        "labels": full_labels,
        "position_ids": full_position_ids,
        "segment_lengths": segment_lengths,
    }

def create_attention_mask(input_ids_blocks):
    assert False, "This method is not used anymore"
    total_len = 0
    for block in input_ids_blocks:
        total_len += len(block)
    
    attn_mask = torch.ones((total_len, total_len), dtype=torch.bool)
    block_end_idx = len(input_ids_blocks[0])
    block_num = 1

    while block_num < len(input_ids_blocks):
        # Worker 1 block
        block_start_idx = block_end_idx
        block_end_idx = block_start_idx + len(input_ids_blocks[block_num])
        attn_mask[block_end_idx:, block_start_idx:block_end_idx] = False
        block_num += 1

        # Worker 2 block
        block_start_idx = block_end_idx
        block_end_idx = block_start_idx + len(input_ids_blocks[block_num])
        attn_mask[block_end_idx:, block_start_idx:block_end_idx] = False
        block_num += 1

        # Worker 3 block
        block_start_idx = block_end_idx
        block_end_idx = block_start_idx + len(input_ids_blocks[block_num])
        attn_mask[block_end_idx:, block_start_idx:block_end_idx] = False
        block_num += 1

        # All workers block
        block_start_idx = block_end_idx
        block_end_idx = block_start_idx + len(input_ids_blocks[block_num])
        block_num += 1

        # Single thread block
        block_start_idx = block_end_idx
        block_end_idx = block_start_idx + len(input_ids_blocks[block_num])
        block_num += 1
    
    causal_mask = torch.tril(torch.ones((total_len, total_len), dtype=torch.bool))
    attn_mask = torch.logical_and(attn_mask, causal_mask)
    return attn_mask

if __name__ == "__main__":
    # Unit test for parse response
    test_response = "<think>"
    test_response += "abc"
    test_response += "<spawn_workers>"
    test_response += "<worker_1>"
    test_response += "def"
    test_response += "</worker_1>"
    test_response += "<worker_2>"
    test_response += "ghi"
    test_response += "</worker_2>"
    test_response += "<worker_3>"
    test_response += "jkl"
    test_response += "</worker_3>"
    test_response += "</spawn_workers>"
    test_response += "mno"
    test_response += "<spawn_workers>"
    test_response += "<worker_1>"
    test_response += "pqr"
    test_response += "</worker_1>"
    test_response += "<worker_2>"
    test_response += "stu"
    test_response += "</worker_2>"
    test_response += "<worker_3>"
    test_response += "vwx"
    test_response += "</worker_3>"
    test_response += "</spawn_workers>"
    test_response += "xyz"
    test_response += "</think>"
    test_response += "<answer>"
    test_response += "abc"
    test_response += "</answer>"
    print(test_response)

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    input_ids_parts, position_ids_parts = parse_response(test_response, tokenizer)
    print("================================================")
    print("Input IDs Parts")
    for token_part, pos_part in zip(input_ids_parts, position_ids_parts):
        print(tokenizer.decode(token_part), len(token_part), pos_part)
        assert len(token_part) == len(pos_part)
    
    print("================================================")
    print("Attention Mask")
    attn_mask = create_attention_mask(input_ids_parts)
    full_input_ids = process_completion(test_response, tokenizer)["input_ids"]
    for i in range(len(full_input_ids)):
        attn_mask_prefix = attn_mask[i, :(i + 1)]
        input_ids_prefix = full_input_ids[:(i + 1)]
        attended_token_list = (['' if not mask else tokenizer.decode([id]) for id, mask in zip(input_ids_prefix, attn_mask_prefix)])
        print(''.join(attended_token_list) + " (attended by) " + tokenizer.decode(full_input_ids[i], skip_special_tokens=False))