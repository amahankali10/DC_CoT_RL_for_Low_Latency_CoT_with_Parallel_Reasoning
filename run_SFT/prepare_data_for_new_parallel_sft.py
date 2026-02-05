from parse_claude_completions_for_new_parallel_generation import process_completion, IGNORE_INDEX

import dataclasses
import torch
from torch.utils.data import Dataset
import transformers
from transformers.utils import logging
logger = logging.get_logger(__name__)

prefix = "<｜begin▁of▁sentence｜><｜User｜>"
suffix = "<｜Assistant｜>"
instruction = "Let's think step by step and output the final answer within \\boxed{{}}."
deepseek_r1_prompt = prefix + "{question} " + instruction + suffix

parallel_instruction = "You can spawn multiple workers to solve this problem in parallel." \
                       " The workers' thoughts are enclosed within <spawn_workers></spawn_workers> tags, and each worker's" \
                       " thought is enclosed within <worker_i></worker_i> tags, where i is the worker number, i.e." \
                       " <spawn_workers><worker_1>worker 1's thought</worker_1><worker_2>worker 2's thought</worker_2>..." \
                       "</spawn_workers>."
deepseek_r1_prompt_parallel = prefix + "{question} " + parallel_instruction + " " + instruction + suffix

def process_example(example, tokenizer, use_parallel_prompt):
    if use_parallel_prompt:
        prompt_template = deepseek_r1_prompt_parallel
    else:
        prompt_template = deepseek_r1_prompt
    prompt = prompt_template.format(question=example["problem"])
    answer = example["parallel_rewritten_completion"]

    # Tokenize the prompt
    tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=False)
    answer_tokenize_dict = process_completion(
        response=answer,
        tokenizer=tokenizer,
        prompt_token_ids=tokenized_prompt,
    )
    return answer_tokenize_dict

def measure_parallel_length_from_segments(segment_lengths):
    parallel_len = 0
    response_len = 0
    block_num = 0

    while block_num < len(segment_lengths):
        # Single thread block
        parallel_len += segment_lengths[block_num]
        response_len += segment_lengths[block_num]
        block_num += 1
        if block_num >= len(segment_lengths):
            break

        # Worker segments
        max_worker = 0
        for _ in range(3):
            response_len += segment_lengths[block_num]
            max_worker = max(max_worker, segment_lengths[block_num])
            block_num += 1
        parallel_len += max_worker

        # Skip all workers block
        block_num += 1
    
    return {
        "parallel_len": parallel_len,
        "response_len": response_len,
    }

class ParallelWorkersSFTDataset(Dataset):

    def __init__(self, train_data, tokenizer, use_parallel_prompt):
        super().__init__()
        self.input_ids = []
        self.labels = []
        self.position_ids = []
        self.segment_lengths = []
        for example in train_data:
            tokenized_example = process_example(example, tokenizer, use_parallel_prompt)
            self.input_ids.append(tokenized_example["input_ids"])
            self.labels.append(tokenized_example["labels"])
            self.position_ids.append(tokenized_example["position_ids"])
            self.segment_lengths.append(tokenized_example["segment_lengths"])
        
        print(f"Max token length: {max(len(input_ids) for input_ids in self.input_ids)}")

        # Average response length
        parallel_lengths = []
        response_lengths = []
        for segment_lengths in self.segment_lengths:
            length_dict = measure_parallel_length_from_segments(segment_lengths)
            parallel_lengths.append(length_dict["parallel_len"])
            response_lengths.append(length_dict["response_len"])
        print(f"Average parallel length: {sum(parallel_lengths) / len(parallel_lengths)}")
        print(f"Average response length: {sum(response_lengths) / len(response_lengths)}")
    
    def remove_long_examples(self, max_length):
        new_input_ids = []
        new_labels = []
        new_position_ids = []
        new_segment_lengths = []
        for input_ids, labels, position_ids, segment_lengths in zip(self.input_ids, self.labels, self.position_ids, self.segment_lengths):
            if len(input_ids) > max_length:
                continue
            new_input_ids.append(input_ids)
            new_labels.append(labels)
            new_position_ids.append(position_ids)
            new_segment_lengths.append(segment_lengths)
        
        self.input_ids = new_input_ids
        self.labels = new_labels
        self.position_ids = new_position_ids
        self.segment_lengths = new_segment_lengths

    def __len__(self):
        assert len(self.input_ids) == len(self.labels) == len(self.position_ids) == len(self.segment_lengths)
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "position_ids": self.position_ids[idx],
            "segment_lengths": self.segment_lengths[idx]
        }

@dataclasses.dataclass
class DataCollatorForParallelWorkerSFT:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids = [torch.tensor(instance["input_ids"]) for instance in instances]
        labels = [torch.tensor(instance["labels"]) for instance in instances]
        position_ids = [torch.tensor(instance["position_ids"]) for instance in instances]
        segment_lengths = [torch.tensor(instance["segment_lengths"]) for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        position_ids = torch.nn.utils.rnn.pad_sequence(position_ids, batch_first=True, padding_value=0)
        segment_lengths = torch.nn.utils.rnn.pad_sequence(segment_lengths, batch_first=True, padding_value=-1)   

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "segment_lengths": segment_lengths,
        }