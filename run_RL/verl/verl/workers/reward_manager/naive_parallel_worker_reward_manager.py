# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch
import math
import statistics

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

###### Function to check the format
worker_start_strings = ["<worker_1>", "<worker_2>", "<worker_3>"]
worker_end_strings = ["</worker_1>", "</worker_2>", "</worker_3>"]
worker_block_start = "<spawn_workers>"
worker_block_end = "</spawn_workers>"
think_start = "<think>"
think_end = "</think>"
answer_start = "<answer>"
answer_end = "</answer>"

def check_parallel_worker_format(completion: str) -> bool:
    """
    The format of the completion is as follows:
    <think>
    <spawn_workers>
    <worker_1> some text here </worker_1>
    <worker_2> some text here </worker_2>
    <worker_3> some text here </worker_3>
    </spawn_workers>
    some text here.
    <spawn_workers>
    <worker_1> some text here </worker_1>
    <worker_2> some text here </worker_2>
    <worker_3> some text here </worker_3>
    </spawn_workers>
    some text here.
    ... an arbitrary number of times ...
    some text here
    </think>
    <answer>
    some text here
    </answer>

    Notes:
    - The <think> and <answer> tags must be present.
    - The <think></think> and <answer></answer> blocks must be disjoint.
    - Within the <spawn_workers></spawn_workers> blocks, there should not be any text outside of the <worker_1></worker_1>, <worker_2></worker_2>, <worker_3></worker_3> blocks.
    - There should not be any text that is not within the <think></think> or <answer></answer> blocks.
    - Worker blocks must appear in order (1,2,3) and each worker tag must appear exactly once in each <spawn_workers></spawn_workers> block.
    """
    # Check if both think and answer blocks are present
    if think_start not in completion or think_end not in completion:
        return False
    if answer_start not in completion or answer_end not in completion:
        return False
    
    # Check if think and answer blocks are properly ordered and disjoint
    think_start_idx = completion.find(think_start)
    think_end_idx = completion.find(think_end)
    answer_start_idx = completion.find(answer_start)
    answer_end_idx = completion.find(answer_end)
    
    if not (think_start_idx < think_end_idx < answer_start_idx < answer_end_idx):
        return False
    
    # Check that there's no text outside think and answer blocks
    text_before_think = completion[:think_start_idx].strip()
    text_between_blocks = completion[think_end_idx + len(think_end):answer_start_idx].strip()
    text_after_answer = completion[answer_end_idx + len(answer_end):].strip()
    
    if text_before_think or text_between_blocks or text_after_answer:
        return False
    
    # Extract the think block content
    think_content = completion[think_start_idx + len(think_start):think_end_idx].strip()
    
    # Check if there's at least one spawn_workers block in the think content
    if worker_block_start not in think_content or worker_block_end not in think_content:
        return False
    
    # Check each spawn_workers block
    while worker_block_start in think_content:
        # Find the next spawn_workers block
        block_start = think_content.find(worker_block_start)
        block_end = think_content.find(worker_block_end, block_start)
        
        if block_end == -1:  # No matching end tag
            return False
        
        # Extract the content before the block - we enforce that it
        # is non-empty to prevent the model to go directly into spawning
        # workers without thinking/assigning subtasks to workers.
        pre_block_content = think_content[:block_start].strip()
        if not pre_block_content:
            return False
            
        # Extract the block content and remove the opening tag
        block_content = think_content[block_start + len(worker_block_start):block_end].strip()
        
        # Process each worker in order
        for worker_idx in range(3):
            worker_start = worker_start_strings[worker_idx]
            worker_end = worker_end_strings[worker_idx]
            
            # Check if this worker tag exists at the start of the remaining content
            if not block_content.startswith(worker_start):
                return False
                
            # Find the end of this worker block
            worker_end_idx = block_content.find(worker_end)
            if worker_end_idx == -1:
                return False
                
            # Remove this worker block and any whitespace
            block_content = block_content[worker_end_idx + len(worker_end):].strip()
        
        # After processing all workers, block_content should be empty
        if block_content:
            return False
        
        # Move to next spawn_workers block
        think_content = think_content[block_end + len(worker_block_end):].strip()
    
    return True

def uses_think(completion: str) -> bool:
    if think_start not in completion or think_end not in completion:
        return False
    
    think_start_idx = completion.find(think_start)
    think_end_idx = completion.find(think_end)

    if not (think_start_idx < think_end_idx):
        return False

    text_before_think = completion[:think_start_idx].strip()
    if text_before_think:
        return False
    
    return True

# This is the estimator, for a single problem, of that problem's pass rate.
# Obtained from https://arxiv.org/abs/2506.01347 - this is then averaged over all the problems.
def pass_at_k_estimator(num_samples, num_correct, k):
    from scipy.special import comb
    numerator = comb(num_samples - num_correct, k, exact=True)
    denominator = comb(num_samples, k, exact=True)
    return 1 - numerator / denominator

@register("naive_parallel_worker")
class NaiveRewardManagerForParallelWorkerRollout:
    def __init__(self, tokenizer, num_examine, compute_score=None, length_penalty_coeff=None, correct_and_no_format_reward=0.0, format_and_incorrect_reward=0.0, reward_fn_key="data_source", use_deepscaler_reward=False, length_penalty_cutoff_length=None, enforce_answer_between_answer_tags=False, prepend_think_token_response=False) -> None:
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.length_penalty_coeff = length_penalty_coeff
        self.correct_and_no_format_reward = correct_and_no_format_reward
        self.format_and_incorrect_reward = format_and_incorrect_reward
        assert correct_and_no_format_reward is not None
        assert format_and_incorrect_reward is not None
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

        # We bucket responses by length for logging purposes
        self.bucket_size = 4000
        self.max_bucket = 16000

        self.use_deepscaler_reward = use_deepscaler_reward
        if self.use_deepscaler_reward:
            assert self.correct_and_no_format_reward == 1.0
            assert self.format_and_incorrect_reward == 0.0
        
        self.enforce_answer_between_answer_tags = enforce_answer_between_answer_tags
        if self.enforce_answer_between_answer_tags:
            assert not self.use_deepscaler_reward
        
        self.length_penalty_cutoff_length = length_penalty_cutoff_length

        self.prepend_think_token_response = prepend_think_token_response

    def __call__(self, data: DataProto, return_dict=True):

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # Set up metrics
        reward_extra_info["score_list"] = []
        reward_extra_info["math_verify_score_list"] = [] # Here we just log the math verify score of the response, regardless of any other considerations.
        reward_extra_info["parallel_length_list"] = []
        reward_extra_info["correct_responses_parallel_length"] = []
        reward_extra_info["incorrect_responses_parallel_length"] = []
        reward_extra_info["response_length_list"] = []
        reward_extra_info["correct_responses_response_length"] = []
        reward_extra_info["incorrect_responses_response_length"] = []
        reward_extra_info["correctness_list"] = []
        reward_extra_info["has_EOS_token_list"] = []
        reward_extra_info["truncated_responses_correctness_list"] = []
        reward_extra_info["truncated_responses_reward_list"] = []
        reward_extra_info["satisfies_format_list"] = []
        reward_extra_info["correct_format_correctness_list"] = []
        reward_extra_info["reaches_answer_tag_list"] = []
        reward_extra_info["has_spawn_workers_list"] = []
        reward_extra_info["has_spawn_workers_correctness_list"] = []
        reward_extra_info["has_spawn_workers_parallel_length_list"] = []
        reward_extra_info["no_spawn_workers_parallel_length_list"] = []
        reward_extra_info["correct_above_length_cutoff_list"] = []
        pass_at_k_uid_dict = dict()
        pass_at_k_math_verify_uid_dict = dict()
        pass_at_k_correct_and_format_uid_dict = dict()
        uid_to_reward_of_correct_resps_dict = dict()

        # Text before delegating
        reward_extra_info["has_word_subtask_list"] = []
        reward_extra_info["has_word_subtask_correctness_list"] = []
        reward_extra_info["NO_word_subtask_correctness_list"] = []
        reward_extra_info["has_word_subtask_parallel_length_list"] = []
        reward_extra_info["NO_word_subtask_parallel_length_list"] = []
        reward_extra_info["has_word_subtask_total_length_list"] = []
        reward_extra_info["NO_word_subtask_total_length_list"] = []
        reward_extra_info["has_text_before_delegate_list"] = []
        reward_extra_info["has_text_before_delegate_correctness_list"] = []
        reward_extra_info["NO_text_before_delegate_correctness_list"] = []
        reward_extra_info["has_text_before_delegate_parallel_length_list"] = []
        reward_extra_info["NO_text_before_delegate_parallel_length_list"] = []
        reward_extra_info["has_text_before_delegate_total_length_list"] = []
        reward_extra_info["NO_text_before_delegate_total_length_list"] = []

        # Degree of parallelism buckets
        step = 0.05
        lval = 1.0
        hval = 3.0
        n_steps = int((hval - lval) / step)
        for i in range(n_steps):
            start = lval + i * step
            end = start + step
            reward_extra_info[f"degree_of_parallelism_bucket_{start}_{end}_list"] = [0.0]
            reward_extra_info[f"degree_of_parallelism_bucket_{start}_{end}_correctness_list"] = []
            reward_extra_info[f"degree_of_parallelism_bucket_{start}_{end}_parallel_length_list"] = []

        if self.use_deepscaler_reward:
            reward_extra_info["has_think"] = []

        # Initialize correctness lists for parallel length buckets
        for start in range(0, self.max_bucket, self.bucket_size):
            end = start + self.bucket_size
            reward_extra_info[f"parallel_response_length_bucket_{start}_{end}_correctness_list"] = []
        # Initialize correctness lists for response length buckets
        for start in range(0, self.max_bucket, self.bucket_size):
            end = start + self.bucket_size
            reward_extra_info[f"response_length_bucket_{start}_{end}_correctness_list"] = []
        
        # Additional parallel metrics
        max_parallel_rounds = 0
        for i in range(len(data)):
            data_item = data[i]
            current_num_parallel_rounds = data_item.batch["num_parallel_rounds"].item()
            max_parallel_rounds = max(max_parallel_rounds, current_num_parallel_rounds)
        reward_extra_info["num_parallel_rounds_list"] = []
        for r in range(max_parallel_rounds + 1):
            reward_extra_info[f"num_parallel_rounds_{r}_list"] = []
        reward_extra_info["degree_of_parallelism_list"] = []
        for r in range(max_parallel_rounds + 1):
            reward_extra_info[f"avg_num_tokens_round_{r}_list"] = []
            reward_extra_info[f"max_num_tokens_round_{r}_list"] = []
            reward_extra_info[f"min_num_tokens_round_{r}_list"] = []
            reward_extra_info[f"max_min_tokens_diff_round_{r}_list"] = []
        reward_extra_info["avg_num_tokens_avg_across_rounds_list"] = []
        reward_extra_info["max_num_tokens_avg_across_rounds_list"] = []
        reward_extra_info["min_num_tokens_avg_across_rounds_list"] = []
        reward_extra_info["max_min_tokens_diff_avg_across_rounds_list"] = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            non_dup_response_ids = data_item.batch["non_dup_worker_response_ids"]
            non_dup_response_str = self.tokenizer.decode(non_dup_response_ids, skip_special_tokens=True)
            if self.prepend_think_token_response:
                non_dup_response_str = "<think>\n" + non_dup_response_str
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            if self.use_deepscaler_reward:
                if uses_think(non_dup_response_str):
                    ans_str = non_dup_response_str.split(think_end)[1]
                    score = self.compute_score(
                        data_source=data_source,
                        solution_str=ans_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    )
                    reward_extra_info["has_think"].append(1.0)
                else:
                    score = 0.0
                    reward_extra_info["has_think"].append(0.0)
            elif self.enforce_answer_between_answer_tags:
                # In this case, enforce that the correct answer has to appear between <answer></answer>.
                answer_start_idx = non_dup_response_str.find(answer_start)
                answer_end_idx = non_dup_response_str.find(answer_end)
                if answer_start_idx == -1 or answer_end_idx == -1:
                    score = 0.0
                elif answer_start_idx >= answer_end_idx:
                    score = 0.0
                else:
                    ans_str = non_dup_response_str[answer_start_idx + len(answer_start):answer_end_idx]
                    score = self.compute_score(
                        data_source=data_source,
                        solution_str=ans_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    )
            else:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=non_dup_response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
            
            math_verify_score = self.compute_score(
                data_source=data_source,
                solution_str=non_dup_response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            reward_extra_info["math_verify_score_list"].append(math_verify_score)

            current_uid = data_item.non_tensor_batch["uid"]
            if current_uid not in pass_at_k_math_verify_uid_dict:
                pass_at_k_math_verify_uid_dict[current_uid] = []
            if abs(math_verify_score - 1.0) < 1e-4:
                pass_at_k_math_verify_uid_dict[current_uid].append(1.0)
            else:
                pass_at_k_math_verify_uid_dict[current_uid].append(0.0)

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            
            is_correct = abs(reward - 1.0) < 1e-4

            reward, reward_extra_info, pass_at_k_uid_dict = self.update_metrics(
                metrics_dict=reward_extra_info,
                pass_at_k_uid_dict=pass_at_k_uid_dict,
                data_item=data_item,
                correctness_score=reward,
                response_str=non_dup_response_str,
                prompt_length=prompt_length,
            )

            reward_extra_info = self.add_parallel_metrics(
                metrics_dict=reward_extra_info,
                data_item=data_item,
                max_parallel_rounds=max_parallel_rounds,
            )

            reward_tensor[i, valid_response_length - 1] = reward

            satisfies_format = check_parallel_worker_format(non_dup_response_str)

            if is_correct:
                uid = data_item.non_tensor_batch["uid"]
                if uid not in uid_to_reward_of_correct_resps_dict:
                    uid_to_reward_of_correct_resps_dict[uid] = []
                uid_to_reward_of_correct_resps_dict[uid].append(reward)
            
            current_uid = data_item.non_tensor_batch["uid"]
            if current_uid not in pass_at_k_correct_and_format_uid_dict:
                pass_at_k_correct_and_format_uid_dict[current_uid] = []
            if is_correct and satisfies_format:
                pass_at_k_correct_and_format_uid_dict[current_uid].append(1.0)
            else:
                pass_at_k_correct_and_format_uid_dict[current_uid].append(0.0)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", non_dup_response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
                print("[reward]", reward)
        
        # Add pass@k lists to metrics_dict
        for uid, bool_list in pass_at_k_uid_dict.items():
            num_attempts = len(bool_list)
            for k in range(1, num_attempts + 1):
                current_key = f"pass_at_{k}_list"
                if current_key not in reward_extra_info:
                    reward_extra_info[current_key] = []
                
                num_correct = sum([int(b) for b in bool_list])
                num_samples = len(bool_list)
                current_metric = pass_at_k_estimator(num_samples, num_correct, k)
                reward_extra_info[current_key].append(current_metric)
        
        for uid, bool_list in pass_at_k_math_verify_uid_dict.items():
            num_attempts = len(bool_list)
            for k in range(1, num_attempts + 1):
                current_key = f"math_verify_pass_at_{k}_list"
                if current_key not in reward_extra_info:
                    reward_extra_info[current_key] = []
                
                num_correct = sum([int(b) for b in bool_list])
                num_samples = len(bool_list)
                current_metric = pass_at_k_estimator(num_samples, num_correct, k)
                reward_extra_info[current_key].append(current_metric)
        
        for uid, bool_list in pass_at_k_correct_and_format_uid_dict.items():
            num_attempts = len(bool_list)
            for k in range(1, num_attempts + 1):
                current_key = f"correct_and_format_pass_at_{k}_list"
                if current_key not in reward_extra_info:
                    reward_extra_info[current_key] = []
                
                num_correct = sum([int(b) for b in bool_list])
                num_samples = len(bool_list)
                current_metric = pass_at_k_estimator(num_samples, num_correct, k)
                reward_extra_info[current_key].append(current_metric)

        median_reward_of_correct_resps = dict()
        for uid in uid_to_reward_of_correct_resps_dict.keys():
            correct_resp_reward_list = uid_to_reward_of_correct_resps_dict[uid]
            assert len(correct_resp_reward_list) > 0
            median_reward_of_correct_resps[uid] = statistics.median(correct_resp_reward_list)
        
        reward_extra_info["median_reward_of_correct_resps_list"] = []
        for uid in median_reward_of_correct_resps.keys():
            reward_extra_info["median_reward_of_correct_resps_list"].append(median_reward_of_correct_resps[uid])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "median_reward_of_correct_resps_per_uid": median_reward_of_correct_resps,
                "uid_to_reward_of_correct_resps_dict": uid_to_reward_of_correct_resps_dict,
            }
        else:
            assert False, "This should not be called"
            return reward_tensor

    def update_metrics(self, 
                       metrics_dict: dict, 
                       pass_at_k_uid_dict: dict, 
                       data_item, 
                       correctness_score, 
                       response_str,
                       prompt_length,
                       is_dapo=False):
        """
        correctness_score should be the original score returned by self.compute_score
        prompt_length is the shape of the prompt tensor.
        response_str is the response string with worker tokens only present once.

        This method updates the metrics dict, and also returns the updated score based on:
        - length penalty
        - format reward

        Returns
        - new metrics dict
        - new score
        - new pass@k UID dict
        """
        has_eos_token = self.tokenizer.eos_token_id in data_item.batch["responses"]
        if has_eos_token:
            metrics_dict["has_EOS_token_list"].append(1.0)
        else:
            metrics_dict["has_EOS_token_list"].append(0.0)

        is_correct_response = abs(correctness_score - 1.0) < 1e-6
        if is_correct_response:
            metrics_dict["correctness_list"].append(1.0)
        else:
            metrics_dict["correctness_list"].append(0.0)
        
        if worker_block_start in response_str:
            metrics_dict["has_spawn_workers_list"].append(1.0)
            if is_correct_response:
                metrics_dict["has_spawn_workers_correctness_list"].append(1.0)
            else:
                metrics_dict["has_spawn_workers_correctness_list"].append(0.0)
        else:
            metrics_dict["has_spawn_workers_list"].append(0.0)
        
        # Log correctness if responses is truncated
        if not has_eos_token:
            if is_correct_response:
                metrics_dict["truncated_responses_correctness_list"].append(1.0)
            else:
                metrics_dict["truncated_responses_correctness_list"].append(0.0)
        
        satisfies_format = check_parallel_worker_format(response_str)
        if is_correct_response and satisfies_format:
            score = correctness_score
        elif is_correct_response and not satisfies_format:
            score = self.correct_and_no_format_reward
        elif not is_correct_response and satisfies_format:
            score = self.format_and_incorrect_reward
        else:
            score = 0.0
        metrics_dict["score_list"].append(score)

        # Log whether format is satisfied
        if satisfies_format:
            metrics_dict["satisfies_format_list"].append(1.0)
        else:
            metrics_dict["satisfies_format_list"].append(0.0)
        
        if satisfies_format:
            if is_correct_response:
                metrics_dict["correct_format_correctness_list"].append(1.0)
            else:
                metrics_dict["correct_format_correctness_list"].append(0.0)
        
        # Update dict used for pass@k
        # Measuring only correctness now
        uid = data_item.non_tensor_batch["uid"]
        if uid not in pass_at_k_uid_dict:
            pass_at_k_uid_dict[uid] = []
        if is_correct_response:
            pass_at_k_uid_dict[uid].append(1.0)
        else:
            pass_at_k_uid_dict[uid].append(0.0)
        
        # UPDATE SCORE WITH LENGTH PENALTY
        # ONLY PENALIZE CORRECT RESPONSES - INCORRECT RESPONSES ARE NOT REWARDED FOR BEING SHORT
        # Measure parallel length
        response_parallel_length = data_item.batch["parallel_lengths"].item()
        if is_correct_response:
            if is_dapo:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                cutoff = self.max_resp_len - overlong_buffer_len
                if response_parallel_length > cutoff:
                    metrics_dict["correct_above_length_cutoff_list"].append(1.0)
                else:
                    metrics_dict["correct_above_length_cutoff_list"].append(0.0)
            else:
                # Calculate length penalty
                score = score - self.length_penalty_coeff * max(0, response_parallel_length - self.length_penalty_cutoff_length)
                if response_parallel_length > self.length_penalty_cutoff_length:
                    metrics_dict["correct_above_length_cutoff_list"].append(1.0)
                else:
                    metrics_dict["correct_above_length_cutoff_list"].append(0.0)
        
        metrics_dict["parallel_length_list"].append(response_parallel_length)
        if is_correct_response:
            metrics_dict["correct_responses_parallel_length"].append(response_parallel_length)
        else:
            metrics_dict["incorrect_responses_parallel_length"].append(response_parallel_length)
        
        if worker_block_start in response_str:
            metrics_dict["has_spawn_workers_parallel_length_list"].append(response_parallel_length)
        else:
            metrics_dict["no_spawn_workers_parallel_length_list"].append(response_parallel_length)

        # Bucket correctness per parallel length bucket
        length = int(response_parallel_length)
        for start in range(0, self.max_bucket, self.bucket_size):
            end = start + self.bucket_size
            if start <= length < end:
                metrics_dict[f"parallel_response_length_bucket_{start}_{end}_correctness_list"].append(1.0 if is_correct_response else 0.0)
                break

        # Log score if response is truncated
        if not has_eos_token:
            metrics_dict["truncated_responses_reward_list"].append(score)

        # Log whether response has <answer></answer> tags
        if answer_start in response_str and answer_end in response_str:
            metrics_dict["reaches_answer_tag_list"].append(1.0)
        else:
            metrics_dict["reaches_answer_tag_list"].append(0.0)
        
        # Measure number of tokens in response
        non_dup_response_length = data_item.batch["response_lengths"]
        metrics_dict["response_length_list"].append(non_dup_response_length)
        if is_correct_response:
            metrics_dict["correct_responses_response_length"].append(non_dup_response_length)
        else:
            metrics_dict["incorrect_responses_response_length"].append(non_dup_response_length)

        # Bucket correctness per response length bucket
        length = int(non_dup_response_length)
        for start in range(0, self.max_bucket, self.bucket_size):
            end = start + self.bucket_size
            if start <= length < end:
                metrics_dict[f"response_length_bucket_{start}_{end}_correctness_list"].append(1.0 if is_correct_response else 0.0)
                break    

        # Update text before delegate metrics
        if satisfies_format:
            think_start_idx = response_str.find(think_start)
            spawn_workers_start_idx = response_str.find(worker_block_start)
            text_before_delegate = response_str[think_start_idx + len(think_start):spawn_workers_start_idx].strip()

            if text_before_delegate:
                metrics_dict["has_text_before_delegate_list"].append(1.0)
                metrics_dict["has_text_before_delegate_correctness_list"].append(1.0 if is_correct_response else 0.0)
                metrics_dict["has_text_before_delegate_parallel_length_list"].append(response_parallel_length)
                metrics_dict["has_text_before_delegate_total_length_list"].append(non_dup_response_length)
            else:
                metrics_dict["has_text_before_delegate_list"].append(0.0)
                metrics_dict["NO_text_before_delegate_correctness_list"].append(1.0 if is_correct_response else 0.0)
                metrics_dict["NO_text_before_delegate_parallel_length_list"].append(response_parallel_length)
                metrics_dict["NO_text_before_delegate_total_length_list"].append(non_dup_response_length)
            
            if "subtask" in text_before_delegate:
                metrics_dict["has_word_subtask_list"].append(1.0)
                metrics_dict["has_word_subtask_correctness_list"].append(1.0 if is_correct_response else 0.0)
                metrics_dict["has_word_subtask_parallel_length_list"].append(response_parallel_length)
                metrics_dict["has_word_subtask_total_length_list"].append(non_dup_response_length)
            else:
                metrics_dict["has_word_subtask_list"].append(0.0)
                metrics_dict["NO_word_subtask_correctness_list"].append(1.0 if is_correct_response else 0.0)
                metrics_dict["NO_word_subtask_parallel_length_list"].append(response_parallel_length)
                metrics_dict["NO_word_subtask_total_length_list"].append(non_dup_response_length)
        
        # Degree of parallelism histogram
        degree_of_parallelism = non_dup_response_length / response_parallel_length
        step = 0.05
        lval = 1.0
        hval = 3.0
        n_steps = int((hval - lval) / step)
        for i in range(n_steps):
            start = lval + i * step
            end = start + step
            if start <= degree_of_parallelism < end:
                metrics_dict[f"degree_of_parallelism_bucket_{start}_{end}_list"][0] += 1.0
                metrics_dict[f"degree_of_parallelism_bucket_{start}_{end}_correctness_list"].append(1.0 if is_correct_response else 0.0)
                metrics_dict[f"degree_of_parallelism_bucket_{start}_{end}_parallel_length_list"].append(response_parallel_length)

        return score, metrics_dict, pass_at_k_uid_dict
    
    def add_parallel_metrics(self, metrics_dict, data_item, max_parallel_rounds):
        # Update num parallel rounds
        num_parallel_rounds = data_item.batch["num_parallel_rounds"].item()
        metrics_dict["num_parallel_rounds_list"].append(num_parallel_rounds)
        for r in range(max_parallel_rounds + 1):
            key = f"num_parallel_rounds_{r}_list"
            if r == num_parallel_rounds:
                metrics_dict[key].append(1.0)
            else:
                metrics_dict[key].append(0.0)
        
        # Update degree of parallelism
        parallel_len = data_item.batch["parallel_lengths"].item()
        response_len = data_item.batch["response_lengths"].item()
        degree_of_parallelism = response_len / parallel_len
        metrics_dict["degree_of_parallelism_list"].append(degree_of_parallelism)

        # Extract avg, max, min lengths for the relevant rounds
        avg_round_tokens = data_item.batch["avg_round_tokens"][:num_parallel_rounds].cpu().tolist()
        max_round_tokens = data_item.batch["max_round_tokens"][:num_parallel_rounds].cpu().tolist()
        min_round_tokens = data_item.batch["min_round_tokens"][:num_parallel_rounds].cpu().tolist()
        max_min_diff = [max_round_tokens[i] - min_round_tokens[i] for i in range(num_parallel_rounds)]

        # Update avg, max, min, max_min_diff for the relevant rounds
        for r in range(num_parallel_rounds):
            avg_key = f"avg_num_tokens_round_{r}_list"
            metrics_dict[avg_key].append(avg_round_tokens[r])

            max_key = f"max_num_tokens_round_{r}_list"
            metrics_dict[max_key].append(max_round_tokens[r])

            min_key = f"min_num_tokens_round_{r}_list"
            metrics_dict[min_key].append(min_round_tokens[r])

            max_min_diff_key = f"max_min_tokens_diff_round_{r}_list"
            metrics_dict[max_min_diff_key].append(max_min_diff[r])
        
        # Update all 4 statistics averaged across rounds
        if num_parallel_rounds > 0:
            avg_of_avg_across_rounds = sum(avg_round_tokens) / num_parallel_rounds
            metrics_dict["avg_num_tokens_avg_across_rounds_list"].append(avg_of_avg_across_rounds)

            avg_of_max_across_rounds = sum(max_round_tokens) / num_parallel_rounds
            metrics_dict["max_num_tokens_avg_across_rounds_list"].append(avg_of_max_across_rounds)

            avg_of_min_across_rounds = sum(min_round_tokens) / num_parallel_rounds
            metrics_dict["min_num_tokens_avg_across_rounds_list"].append(avg_of_min_across_rounds)

            avg_of_max_min_diff_across_rounds = sum(max_min_diff) / num_parallel_rounds
            metrics_dict["max_min_tokens_diff_avg_across_rounds_list"].append(avg_of_max_min_diff_across_rounds)

        return metrics_dict