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
import numpy as np
import torch
import statistics

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

import random

from verl.workers.reward_manager.naive_parallel_worker_reward_manager import (
    uses_think,
    check_parallel_worker_format, 
    pass_at_k_estimator,
    NaiveRewardManagerForParallelWorkerRollout,
    think_end,
    answer_start,
    answer_end,
)

@register("dapo_parallel_worker")
class DAPOParallelWorkerRewardManager(NaiveRewardManagerForParallelWorkerRollout):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, length_penalty_coeff=0.0, correct_and_no_format_reward=0.0, format_and_incorrect_reward=0.0, reward_fn_key="data_source", max_resp_len=None, overlong_buffer_cfg=None, total_tokens_incentive=0.0, use_deepscaler_reward=False, enforce_answer_between_answer_tags=False, prepend_think_token_response=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.length_penalty_coeff = length_penalty_coeff
        self.correct_and_no_format_reward = correct_and_no_format_reward
        self.format_and_incorrect_reward = format_and_incorrect_reward
        assert correct_and_no_format_reward is not None
        assert format_and_incorrect_reward is not None
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.total_tokens_incentive = total_tokens_incentive

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

            max_len_for_overlong = self.overlong_buffer_cfg.get("max_len_for_overlong", -1)
            if max_len_for_overlong > 0:
                self.max_len_for_overlong = max_len_for_overlong
            else:
                self.max_len_for_overlong = self.max_resp_len

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
        reward_extra_info["all_wrong_filter_metric"] = []
        reward_extra_info["all_wrong_or_no_format_filter_metric"] = []
        reward_extra_info["math_verify_plus_format_all_or_none_filter_metric"] = []
        reward_extra_info["overlong_reward"] = []
        reward_extra_info["overlong"] = []
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
            # I believe the following is not needed since EOS token and PAD token are removed with skip_special_tokens=True
            # eos_token = self.tokenizer.eos_token
            # if response_str.endswith(eos_token):
            #     response_str = response_str[: -len(eos_token)]


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
            
            # Update all_wrong_filter_metric
            is_correct = abs(reward - 1.0) < 1e-4
            if is_correct:
                custom_metric = random.random()
                reward_extra_info["all_wrong_filter_metric"].append(custom_metric)
            else:
                reward_extra_info["all_wrong_filter_metric"].append(0.0)
            
            # Update all_wrong_or_no_format_filter_metric
            satisfies_format = check_parallel_worker_format(non_dup_response_str)
            if is_correct and satisfies_format:
                custom_metric = random.random()
                reward_extra_info["all_wrong_or_no_format_filter_metric"].append(custom_metric)
            else:
                reward_extra_info["all_wrong_or_no_format_filter_metric"].append(0.0)
            
            if is_correct and satisfies_format:
                reward_extra_info["math_verify_plus_format_all_or_none_filter_metric"].append(1.0)
            else:
                reward_extra_info["math_verify_plus_format_all_or_none_filter_metric"].append(0.0)

            current_uid = data_item.non_tensor_batch["uid"]
            if current_uid not in pass_at_k_correct_and_format_uid_dict:
                pass_at_k_correct_and_format_uid_dict[current_uid] = []
            if is_correct and satisfies_format:
                pass_at_k_correct_and_format_uid_dict[current_uid].append(1.0)
            else:
                pass_at_k_correct_and_format_uid_dict[current_uid].append(0.0)

            reward, reward_extra_info, pass_at_k_uid_dict = self.update_metrics(
                metrics_dict=reward_extra_info,
                pass_at_k_uid_dict=pass_at_k_uid_dict,
                data_item=data_item,
                correctness_score=reward,
                response_str=non_dup_response_str,
                prompt_length=prompt_length,
                is_dapo=True,
            )

            reward_extra_info = self.add_parallel_metrics(
                metrics_dict=reward_extra_info,
                data_item=data_item,
                max_parallel_rounds=max_parallel_rounds,
            )

            # Overlong reward shaping
            if self.overlong_buffer_cfg.enable:
                if self.overlong_buffer_cfg.get("parallel", False):
                    overlong_buffer_len = self.overlong_buffer_cfg.len
                    expected_len = self.max_len_for_overlong - overlong_buffer_len
                    parallel_len_non_dup = data_item.batch["parallel_lengths"].item()
                    exceed_len = parallel_len_non_dup - expected_len
                    overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                    if self.overlong_buffer_cfg.log:
                        reward_extra_info["overlong_reward"].append(overlong_reward)
                        reward_extra_info["overlong"].append(overlong_reward < 0)
                else:
                    overlong_buffer_len = self.overlong_buffer_cfg.len
                    expected_len = self.max_len_for_overlong - overlong_buffer_len
                    non_dup_response_length = data_item.batch["response_lengths"].item()
                    exceed_len = non_dup_response_length - expected_len
                    overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                    if self.overlong_buffer_cfg.log:
                        reward_extra_info["overlong_reward"].append(overlong_reward)
                        reward_extra_info["overlong"].append(overlong_reward < 0)
                
                if self.overlong_buffer_cfg.get("penalize_only_correct_resps", False):
                    if is_correct:
                        reward += overlong_reward
                elif self.overlong_buffer_cfg.get("penalize_only_resps_with_correct_and_format", False):
                    if is_correct and satisfies_format:
                        reward += overlong_reward
                else:
                    reward += overlong_reward

            reward_tensor[i, valid_response_length - 1] = reward

            if is_correct:
                uid = data_item.non_tensor_batch["uid"]
                if uid not in uid_to_reward_of_correct_resps_dict:
                    uid_to_reward_of_correct_resps_dict[uid] = []
                uid_to_reward_of_correct_resps_dict[uid].append(reward)
                print("Added correct response's reward: ", reward, " for uid: ", uid)

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

        # Compute UID -> parallel length std. dev.
        uid_parallel_length_dict = defaultdict(list)
        for i in range(len(data)):
            uid = data[i].non_tensor_batch["uid"]
            parallel_len = data[i].batch["parallel_lengths"]
            uid_parallel_length_dict[uid].append(parallel_len)
        for uid, parallel_length_list in uid_parallel_length_dict.items():
            std_dev = np.std(parallel_length_list)
            reward_extra_info["parallel_length_std_dev"].append(std_dev)

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