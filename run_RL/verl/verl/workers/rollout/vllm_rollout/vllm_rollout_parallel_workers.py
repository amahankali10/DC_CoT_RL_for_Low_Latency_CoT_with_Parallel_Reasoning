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

import logging
import os
import ast
import gc
from contextlib import contextmanager
from copy import deepcopy
from typing import List
from types import SimpleNamespace

import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch import nn
import vllm
from vllm import TokensPrompt
from vllm import SamplingParams
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# This is the key change to get vllm log
logging.getLogger("vllm").setLevel(logging.INFO)

SPAWN_WORKERS = "<spawn_workers>"
UNSPAWN_WORKERS = "</spawn_workers>"
WORKER_STARTS = ["<worker_1>", "<worker_2>", "<worker_3>"]
WORKER_ENDS = ["</worker_1>", "</worker_2>", "</worker_3>"]

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

class vLLMRolloutParallelWorkers(BaseRollout):

    def get_engine(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank() % torch.cuda.device_count()
        else:
            local_rank = 0
        
        vllm_engine_args = vllm.EngineArgs(
            model=self.base_model_path,
            dtype=self.engine_args.dtype,
            enable_prefix_caching=True,
            block_size=self.engine_args.block_size,
            gpu_memory_utilization=self.engine_args.gpu_memory_utilization,
            max_num_batched_tokens=self.engine_args.max_num_batched_tokens,
            max_num_seqs=self.engine_args.max_num_seqs,
            enforce_eager=self.engine_args.enforce_eager,
            max_model_len=self.engine_args.max_model_len,
            max_seq_len_to_capture=self.engine_args.max_seq_len_to_capture,
            device=f"cuda:{local_rank}",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            distributed_executor_backend="uni",
        )

        engine = vllm.LLMEngine.from_engine_args(vllm_engine_args)
        return engine

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.actor_module = actor_module
        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        if tensor_parallel_size != 1:
            raise NotImplementedError
        
        # Create vllm engine args
        max_model_len = config.prompt_length + 3 * config.response_length
        max_model_len = int(max_model_len)
        max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)
        max_num_batched_tokens = int(max_num_batched_tokens)
        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )
        self.seed = self.config.get("seed", None)
        self.train_N = self.config.n
        self.val_N = self.config.val_kwargs.n
        assert self.seed is not None, "seed must be set"
        block_size = config.get("block_size", None)
        assert block_size is not None, "block_size must be set"
        max_num_seqs = config.get("max_num_seqs", None) # 2000
        assert max_num_seqs is not None, "max_num_seqs must be set"

        # True if the length limit is based on the parallel
        # length, or the total length
        self.length_limit_parallel = config.get("length_limit_parallel", False)

        self.additional_total_length_limit = config.get("additional_total_length_limit", None)
        if self.additional_total_length_limit is not None:
            assert self.length_limit_parallel, "additional_total_length_limit should only be set with parallel length limit"

        engine_args_dict = {
            "dtype": config.dtype,
            "block_size": block_size,
            "seed": self.seed,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": max_num_seqs,
            "enforce_eager": config.enforce_eager,
            "max_model_len": max_model_len,
            "max_seq_len_to_capture": config.get("max_seq_len_to_capture", max_model_len),
        }
        self.engine_args = SimpleNamespace(**engine_args_dict)

        assert "base_model_path" in kwargs, "Need base_model_path for vllm parallel worker rollout"
        self.base_model_path = kwargs["base_model_path"]
        self.pad_token_id = tokenizer.pad_token_id
    
    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))
        
        do_sample = prompts.meta_info.get("do_sample", True)
        if not do_sample:
            raise NotImplementedError
        is_validate = prompts.meta_info.get("validate", False)
        temperature = -1
        if is_validate:
            temperature = self.config.val_kwargs.temperature
        else:
            temperature = self.config.temperature

        # Duplicate prompts and set new seeds for each copy
        current_N = self.train_N if not is_validate else self.val_N
        new_idx_list = []
        new_seeds = []
        for i in range(batch_size):
            current_prompts = [deepcopy(idx_list[i]) for _ in range(current_N)]
            current_seeds = [self.seed + j for j in range(current_N)]
            new_idx_list.extend(current_prompts)
            new_seeds.extend(current_seeds)

        # Get the engine and use most recent parameters
        # Then do generation
        with FSDP.summon_full_params(self.actor_module):
            engine = self.get_engine()
            model_unwrapped = self.actor_module.module
            vllm_model = engine.model_executor.driver_worker.model_runner.model
            def _clean_named_parameters(named_params):
                for name, param in named_params:
                    while "._fsdp_wrapped_module." in name:
                        name = name.replace("._fsdp_wrapped_module.", ".")
                    if name.endswith("._fsdp_wrapped_module"):
                        name = name.replace("._fsdp_wrapped_module", "")
                    yield name, param
            vllm_model.load_weights(_clean_named_parameters(model_unwrapped.named_parameters()))

            # Generate
            if is_validate:
                top_k = self.config.val_kwargs.top_k
                top_p = self.config.val_kwargs.top_p
            else:
                top_k = self.config.top_k
                top_p = self.config.top_p
            old_token_segments, hardcoded_token_masks = self.process_all_prompts(
                engine=engine,
                idx_list=new_idx_list,
                seeds=new_seeds,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Free engine
            del engine.model_executor
            del engine
            gc.collect()
            torch.cuda.empty_cache()
        
        #########
        # Get non-dup-worker responses, and response length/parallel length
        # to help with reward calculation and metrics
        non_dup_worker_response_ids = []
        response_lengths = []
        parallel_lengths = []
        # Extract parallel metrics as well
        spawn_worker_invocations = []
        avg_round_tokens = [] # (bs, spawn_worker_invocations)
        max_round_tokens = [] # (bs, spawn_worker_invocations)
        min_round_tokens = [] # (bs, spawn_worker_invocations)
        for i in range(batch_size * current_N):
            out_dict = self.get_non_worker_duplicate_response_tensor_and_metrics(old_token_segments[i])

            # Use post-processed token segments
            old_token_segments[i] = out_dict["postprocessed_token_segments"]

            # Used for accuracy/length metrics
            non_dup_worker_response_ids.append(out_dict["non_dup_worker_response_ids"])
            response_lengths.append(out_dict["response_length"])
            parallel_lengths.append(out_dict["parallel_length"])

            # Extract parallel metrics
            spawn_worker_invocations.append(out_dict["number_of_parallel_rounds"])
            avg_round_tokens.append(out_dict["avg_round_lengths"])
            max_round_tokens.append(out_dict["max_round_lengths"])
            min_round_tokens.append(out_dict["min_round_lengths"])
        max_non_dup_len = max(len(non_dup_ids) for non_dup_ids in non_dup_worker_response_ids)
        for non_dup_ids in non_dup_worker_response_ids:
            pad_len = max_non_dup_len - len(non_dup_ids)
            if pad_len > 0:
                non_dup_ids.extend([self.pad_token_id] * pad_len)
        non_dup_worker_response_ids = torch.tensor(non_dup_worker_response_ids, dtype=torch.long).to(idx.device)
        response_lengths = torch.tensor(response_lengths, dtype=torch.long).unsqueeze(1).to(idx.device)
        parallel_lengths = torch.tensor(parallel_lengths, dtype=torch.long).unsqueeze(1).to(idx.device)
        assert max_non_dup_len <= 3 * self.config.response_length, "max_non_dup_len is greater than 3 * response_length"
        if max_non_dup_len < 3 * self.config.response_length:
            pad_len = 3 * self.config.response_length - max_non_dup_len
            right_t = torch.full((batch_size * current_N, pad_len), self.pad_token_id, dtype=torch.long).to(idx.device)
            non_dup_worker_response_ids = torch.cat([non_dup_worker_response_ids, right_t], dim=1)
        
        # Assemble parallel metrics into tensors by padding with -1s up to 3 * self.config.response_length
        max_spawn_worker_invocations = max(spawn_worker_invocations)
        assert max_spawn_worker_invocations <= 3 * self.config.response_length, "max_spawn_worker_invocations is greater than 3 * response_length"
        spawn_worker_invocations = torch.tensor(spawn_worker_invocations, dtype=torch.long).unsqueeze(1).to(idx.device)
        new_avg_round_tokens = torch.zeros((batch_size * current_N, 3 * self.config.response_length), dtype=torch.long)
        new_max_round_tokens = torch.zeros((batch_size * current_N, 3 * self.config.response_length), dtype=torch.long)
        new_min_round_tokens = torch.zeros((batch_size * current_N, 3 * self.config.response_length), dtype=torch.long)
        new_avg_round_tokens[:, :] = -1
        new_max_round_tokens[:, :] = -1
        new_min_round_tokens[:, :] = -1
        for i in range(batch_size * current_N):
            # Update avg
            per_response_avgs = avg_round_tokens[i]
            new_avg_round_tokens[i, :len(per_response_avgs)] = torch.tensor(per_response_avgs, dtype=torch.long)

            # Update max
            per_response_maxs = max_round_tokens[i]
            new_max_round_tokens[i, :len(per_response_maxs)] = torch.tensor(per_response_maxs, dtype=torch.long)

            # Update min
            per_response_mins = min_round_tokens[i]
            new_min_round_tokens[i, :len(per_response_mins)] = torch.tensor(per_response_mins, dtype=torch.long)

            assert len(per_response_avgs) == len(per_response_maxs) == len(per_response_mins), "avg, max, and min round lengths must be the same"
            assert len(per_response_avgs) <= 3 * self.config.response_length, "avg round lengths are greater than 3 * response_length"
        new_avg_round_tokens = new_avg_round_tokens.to(idx.device)
        new_max_round_tokens = new_max_round_tokens.to(idx.device)
        new_min_round_tokens = new_min_round_tokens.to(idx.device)
        avg_round_tokens = new_avg_round_tokens
        max_round_tokens = new_max_round_tokens
        min_round_tokens = new_min_round_tokens

        # Postprocess generation results to get training data
        all_examples_token_segments = []
        all_examples_pos_ids_segments = []
        for i in range(batch_size * current_N):
            current_token_segments, current_pos_ids_segments = self.get_token_segments_and_pos_ids(old_token_segments[i])
            all_examples_token_segments.append(current_token_segments)
            all_examples_pos_ids_segments.append(current_pos_ids_segments)
        
        token_segment_lengths = torch.zeros((batch_size * current_N, 3 * self.config.response_length), dtype=torch.long)
        token_segment_lengths[:, :] = -1
        for i in range(batch_size * current_N):
            current_token_segments = all_examples_token_segments[i]
            cts_lengths = [len(segment) for segment in current_token_segments]
            token_segment_lengths[i, :len(cts_lengths)] = torch.tensor(cts_lengths, dtype=torch.long)
            assert len(cts_lengths) <= 3 * self.config.response_length, "token_segment_lengths is greater than 3 * response_length"
        
        #########################################################
        # Final post-processing to form all of the tensors/
        # data for training the actor
        #########################################################

        ############### Form the response tensors ###############
        length_bound = None
        if self.length_limit_parallel:
            length_bound = 6 * self.config.response_length
        else:
            length_bound = 2 * self.config.response_length
        
        # Form the response token IDs from the token segments
        all_responses = []
        for segment_list in all_examples_token_segments:
            current_response = []
            for segment in segment_list:
                current_response.extend(segment)
            all_responses.append(current_response)
        max_response_len = max(len(response) for response in all_responses)
        if max_response_len < length_bound:
            max_response_len = length_bound
        for response in all_responses:
            if len(response) < max_response_len:
                response.extend([self.pad_token_id] * (max_response_len - len(response)))
        all_responses = torch.tensor(all_responses, dtype=torch.long)

        # Form the hardcoded token masks
        max_hardcoded_mask_len = max(len(mask) for mask in hardcoded_token_masks)
        if max_hardcoded_mask_len < length_bound:
            max_hardcoded_mask_len = length_bound
        for mask in hardcoded_token_masks:
            if len(mask) < max_hardcoded_mask_len:
                mask.extend([False] * (max_hardcoded_mask_len - len(mask)))
        hardcoded_token_masks = torch.tensor(hardcoded_token_masks, dtype=torch.bool)

        # Form the response position IDs from the position ID segments
        all_pos_ids = []
        for segment_list in all_examples_pos_ids_segments:
            current_pos_ids = []
            for segment in segment_list:
                current_pos_ids.extend(segment)
            all_pos_ids.append(current_pos_ids)
        for pos_ids in all_pos_ids:
            if len(pos_ids) < max_response_len:
                max_element = max(pos_ids)
                pos_ids.extend([max_element] * (max_response_len - len(pos_ids)))
        all_pos_ids = torch.tensor(all_pos_ids, dtype=torch.long)

        # Cut off at length bound and move to the right device
        all_responses = all_responses[:, :length_bound]
        hardcoded_token_masks = hardcoded_token_masks[:, :length_bound]
        all_pos_ids = all_pos_ids[:, :length_bound]
        all_responses = all_responses.to(idx.device)
        hardcoded_token_masks = hardcoded_token_masks.to(idx.device)
        all_pos_ids = all_pos_ids.to(idx.device)
        token_segment_lengths = token_segment_lengths.to(idx.device)

        ############### Combine responses with prompts ###############
        prompt_padding_mask = prompts.batch["attention_mask"]
        prompt_position_ids = prompts.batch["position_ids"]
        if current_N > 1:
            idx = idx.repeat_interleave(current_N, dim=0)
            prompt_padding_mask = prompt_padding_mask.repeat_interleave(current_N, dim=0)
            prompt_position_ids = prompt_position_ids.repeat_interleave(current_N, dim=0)
        
        # Create final input IDs
        all_input_ids = torch.cat([idx, all_responses], dim=1)

        # Create final position IDs
        position_id_inc = prompt_position_ids[:, [-1]] + 1
        all_pos_ids = all_pos_ids + position_id_inc
        all_pos_ids = torch.cat([prompt_position_ids, all_pos_ids], dim=1)

        # Create final padding mask
        all_padding_mask = (all_input_ids != self.pad_token_id)

        ############### Return DataProto ###############
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": all_responses,
                "input_ids": all_input_ids,
                "attention_mask": all_padding_mask,
                "position_ids": all_pos_ids,
                "non_dup_worker_response_ids": non_dup_worker_response_ids,
                "response_lengths": response_lengths,
                "parallel_lengths": parallel_lengths,
                "token_segment_lengths": token_segment_lengths,
                "prompt_padding_mask": prompt_padding_mask,
                "hardcoded_token_masks": hardcoded_token_masks,
                "num_parallel_rounds": spawn_worker_invocations,
                "avg_round_tokens": avg_round_tokens,
                "max_round_tokens": max_round_tokens,
                "min_round_tokens": min_round_tokens,
            },
            batch_size=batch_size * current_N,
        )
        return DataProto(batch=batch)

    def process_all_prompts(self, engine, idx_list, seeds, temperature, top_k, top_p):
        """
        Input:
        - engine is the newly created vllm engine in generate_sequences
        - idx_list: List of prompts (each prompt is a list of token IDs)
        - seeds: Seed for each prompt (the caller of this method takes care of setting the seed for different copies of a single prompt)

        Output:
        - List of size (batch_size)
            - Each element is a list of lists, where each sublist corresponds to some phase (i.e. single-thread, worker 1, worker 2, worker 3)
        """
        batch_size = len(idx_list)

        # Data structures for parallel generation
        max_parallel_rounds = self.config.get("max_parallel_rounds", None)
        assert max_parallel_rounds is not None, "max_parallel_rounds must be set"
        max_tokens_in_parallel_round = self.config.get("max_tokens_in_parallel_round", None)
        assert max_tokens_in_parallel_round is not None, "max_tokens_in_parallel_round must be set"
        rounds_left_per_prompt = [max_parallel_rounds] * batch_size
        tokens_left_per_prompt = [self.config.response_length] * batch_size
        # response strings
        completions = [""] * batch_size
        # For each example, a list of lists, where each sublist corresponds to some phase (i.e. single-thread, worker 1, worker 2, worker 3, single-thread)
        # Here, we do not yet repeat the worker tokens (i.e. we do not have the "all workers" segments)
        token_segments = [[] for _ in range(batch_size)]
        hardcoded_token_masks = [[] for _ in range(batch_size)]
        requests_per_prompt = [0] * batch_size

        flattened_tokens = [[] for _ in range(batch_size)]
        for i, prompt_tokens in enumerate(idx_list):
            current_sampling_params = SamplingParams(
                n = 1,
                max_tokens = self.config.response_length,
                temperature = temperature,
                seed = seeds[i],
                stop = [SPAWN_WORKERS],
                include_stop_str_in_output = True,
                skip_special_tokens = False,
                top_k = top_k,
                top_p = top_p,
            )
            request_id_dict = {
                "idx_in_batch": i,
                "round": 0,
                "single_thread": True,
            }
            requests_per_prompt[i] += 1

            flattened_tokens[i].extend(prompt_tokens)
            engine.add_request(
                str(request_id_dict),
                TokensPrompt(prompt_token_ids=deepcopy(flattened_tokens[i])),
                current_sampling_params,
            )

        worker_outputs = []
        for _ in range(batch_size):
            num_workers = len(WORKER_STARTS)
            worker_outputs.append([None] * num_workers)      
        
        while engine.has_unfinished_requests():
            current_step_outputs = engine.step()

            for req_output in current_step_outputs:

                if req_output.finished:
                    request_id_dict = ast.literal_eval(req_output.request_id)
                    idx_in_batch = request_id_dict["idx_in_batch"]
                    output_str = req_output.outputs[0].text
                    generated_tokens = list(req_output.outputs[0].token_ids)
                    num_tokens_generated = len(generated_tokens)

                    if request_id_dict["single_thread"]:
                        ####### Check if it is finished due to EOS or SPAWN_WORKERS
                        ####### IF SPAWN_WORKERS, CREATE NEW REQUESTS FOR EACH WORKER
                        completions[idx_in_batch] += output_str
                        flattened_tokens[idx_in_batch].extend(generated_tokens)
                        hardcoded_token_masks[idx_in_batch].extend([True] * num_tokens_generated)
                        new_token_segment = []
                        if len(token_segments[idx_in_batch]) > 0:
                            new_token_segment.extend(self.tokenizer.encode(UNSPAWN_WORKERS, add_special_tokens=False))
                        new_token_segment.extend(generated_tokens)
                        token_segments[idx_in_batch].append(new_token_segment)
                        tokens_left_per_prompt[idx_in_batch] -= num_tokens_generated

                        if tokens_left_per_prompt[idx_in_batch] > 0 and rounds_left_per_prompt[idx_in_batch] > 0 and req_output.outputs[0].stop_reason == SPAWN_WORKERS:

                            for i in range(len(WORKER_STARTS)):
                                worker_outputs[idx_in_batch][i] = None
                                request_id_dict = {
                                    "idx_in_batch": idx_in_batch,
                                    "round": requests_per_prompt[idx_in_batch],
                                    "single_thread": False,
                                    "worker_idx": i,
                                }

                                current_tokenized_prompt = deepcopy(flattened_tokens[idx_in_batch])
                                current_tokenized_prompt.extend(self.tokenizer.encode(WORKER_STARTS[i], add_special_tokens=False))
                                
                                worker_sampling_params = SamplingParams(
                                    n = 1,
                                    temperature = temperature,
                                    seed = seeds[idx_in_batch] + requests_per_prompt[idx_in_batch] + i,
                                    max_tokens = min(
                                        tokens_left_per_prompt[idx_in_batch],
                                        max_tokens_in_parallel_round,
                                    ),
                                    stop = [WORKER_ENDS[i]],
                                    include_stop_str_in_output = True,
                                    skip_special_tokens = False,
                                    top_k = top_k,
                                    top_p = top_p,
                                )
                                engine.add_request(
                                    str(request_id_dict),
                                    TokensPrompt(prompt_token_ids=current_tokenized_prompt),
                                    worker_sampling_params,
                                )
                            requests_per_prompt[idx_in_batch] += 1
                            rounds_left_per_prompt[idx_in_batch] -= 1
                    else:
                        ####### SAVE THE OUTPUT OF THIS WORKER
                        ####### IF ALL WORKERS ARE FINISHED, CREATE A NEW REQUEST FOR SINGLE THREAD
                        worker_idx = request_id_dict["worker_idx"]
                        worker_outputs[idx_in_batch][worker_idx] = req_output

                        if all(worker_outputs[idx_in_batch]):
                            # The tokens limit is in terms of parallel length -
                            # so take the max of the worker output lengths.
                            worker_output_lengths = [len(out.outputs[0].token_ids) for out in worker_outputs[idx_in_batch]]
                            if self.length_limit_parallel:
                                max_worker_output_length = max(worker_output_lengths)
                                tokens_left_per_prompt[idx_in_batch] -= max_worker_output_length
                            else:
                                tokens_left_per_prompt[idx_in_batch] -= sum(worker_output_lengths)

                            ####### CREATE A NEW REQUEST FOR SINGLE THREAD
                            num_worker_tokens_total = 0
                            for i, w_start in enumerate(WORKER_STARTS):
                                completions[idx_in_batch] += w_start + worker_outputs[idx_in_batch][i].outputs[0].text
                                w_start_tokenized = self.tokenizer.encode(w_start, add_special_tokens=False)
                                main_worker_output = worker_outputs[idx_in_batch][i].outputs[0].token_ids
                                new_segment = []
                                new_segment.extend(w_start_tokenized)
                                new_segment.extend(main_worker_output)
                                token_segments[idx_in_batch].append(new_segment)
                                flattened_tokens[idx_in_batch].extend(new_segment)
                                hardcoded_token_masks[idx_in_batch].extend([False] * len(w_start_tokenized))
                                hardcoded_token_masks[idx_in_batch].extend([True] * len(main_worker_output))
                                num_worker_tokens_total += len(w_start_tokenized) + len(main_worker_output)
                            hardcoded_token_masks[idx_in_batch].extend([False] * num_worker_tokens_total)

                            completions[idx_in_batch] += UNSPAWN_WORKERS # Add this to token_segments later as part of one single-thread segment
                            unspawn_workers_tokenized = self.tokenizer.encode(UNSPAWN_WORKERS, add_special_tokens=False)
                            flattened_tokens[idx_in_batch].extend(unspawn_workers_tokenized)
                            hardcoded_token_masks[idx_in_batch].extend([False] * len(unspawn_workers_tokenized))
                            request_id_dict = {
                                "idx_in_batch": idx_in_batch,
                                "round": requests_per_prompt[idx_in_batch],
                                "single_thread": True,
                            }
                            requests_per_prompt[idx_in_batch] += 1
                            if tokens_left_per_prompt[idx_in_batch] > 0:
                                current_sampling_params = SamplingParams(
                                    n = 1,
                                    temperature = temperature,
                                    seed = seeds[idx_in_batch] + requests_per_prompt[idx_in_batch],
                                    max_tokens = tokens_left_per_prompt[idx_in_batch],
                                    stop = [SPAWN_WORKERS],
                                    include_stop_str_in_output = True,
                                    skip_special_tokens = False,
                                    top_k = top_k,
                                    top_p = top_p,
                                )
                                engine.add_request(
                                    str(request_id_dict),
                                    TokensPrompt(prompt_token_ids=deepcopy(flattened_tokens[idx_in_batch])),
                                    current_sampling_params
                                )
        
        return token_segments, hardcoded_token_masks
    
    # Given token segments returned by process_all_prompts,
    # create a tensor, and compute the response length/parallel lengths.
    # This is because when decoding the response to compute the reward,
    # we want a version where the worker tokens are not duplicated.
    # Here, example_token_segments is a list of lists - the overall list
    # is just for a single example, and each sublist corresponds to some phase.
    def get_non_worker_duplicate_response_tensor_and_metrics_parallel_limit(self, example_token_segments):
        response_ids = []
        response_length = 0
        parallel_length = 0
        budget = self.config.response_length

        # Additional parallel metrics
        number_of_parallel_rounds = 0
        avg_round_lengths = []
        max_round_lengths = []
        min_round_lengths = []

        # postprocessed token segments
        # In case the token segments exceed the budget,
        # we deal with that case here.
        postprocessed_token_segments = []

        idx = 0
        while idx < len(example_token_segments) and budget > 0:
            # Single thread segment
            current_segment = example_token_segments[idx]
            if len(current_segment) > budget:
                current_segment = current_segment[:budget]
            response_ids.extend(current_segment)
            postprocessed_token_segments.append(current_segment)
            response_length += len(current_segment)
            parallel_length += len(current_segment)
            budget -= len(current_segment)
            idx += 1
            if idx >= len(example_token_segments):
                # Last single thread segment
                break
            if budget <= 0:
                break

            # Worker 1
            w1_segment = example_token_segments[idx]
            if len(w1_segment) >= budget:
                w1_segment = w1_segment[:budget]
            response_ids.extend(w1_segment)
            postprocessed_token_segments.append(w1_segment)
            response_length += len(w1_segment)
            idx += 1

            # Worker 2
            w2_segment = example_token_segments[idx]
            if len(w2_segment) > budget:
                w2_segment = w2_segment[:budget]
            response_ids.extend(w2_segment)
            postprocessed_token_segments.append(w2_segment)
            response_length += len(w2_segment)
            idx += 1

            # Worker 3
            w3_segment = example_token_segments[idx]
            if len(w3_segment) > budget:
                w3_segment = w3_segment[:budget]
            response_ids.extend(w3_segment)
            postprocessed_token_segments.append(w3_segment)
            response_length += len(w3_segment)
            idx += 1

            # Update parallel length and budget
            worker_length = max(len(w1_segment), len(w2_segment))
            worker_length = max(worker_length, len(w3_segment))
            parallel_length += worker_length
            budget -= worker_length

            # Update additional parallel metrics
            number_of_parallel_rounds += 1
            current_avg_length = (len(w1_segment) + len(w2_segment) + len(w3_segment)) / 3
            avg_round_lengths.append(current_avg_length)
            current_max_length = worker_length
            max_round_lengths.append(current_max_length)
            current_min_length = min(len(w1_segment), len(w2_segment))
            current_min_length = min(current_min_length, len(w3_segment))
            min_round_lengths.append(current_min_length)
        
        return {
            "non_dup_worker_response_ids": response_ids,
            "response_length": response_length,
            "parallel_length": parallel_length,
            "number_of_parallel_rounds": number_of_parallel_rounds,
            "avg_round_lengths": avg_round_lengths,
            "max_round_lengths": max_round_lengths,
            "min_round_lengths": min_round_lengths,
            "postprocessed_token_segments": postprocessed_token_segments,
        }
    
    def get_non_worker_duplicate_response_tensor_and_metrics_total_limit(self, example_token_segments):
        response_ids = []
        response_length = 0
        parallel_length = 0
        budget = self.config.response_length

        # Additional parallel metrics
        number_of_parallel_rounds = 0
        avg_round_lengths = []
        max_round_lengths = []
        min_round_lengths = []

        # postprocessed token segments
        postprocessed_token_segments = []

        idx = 0
        while idx < len(example_token_segments) and budget > 0:
            # Single thread segment
            current_segment = example_token_segments[idx]
            if len(current_segment) > budget:
                current_segment = current_segment[:budget]
            response_ids.extend(current_segment)
            postprocessed_token_segments.append(current_segment)
            response_length += len(current_segment)
            parallel_length += len(current_segment)
            budget -= len(current_segment)
            idx += 1
            if idx >= len(example_token_segments):
                break
            if budget <= 0:
                break

            # Worker 1
            w1_segment = example_token_segments[idx]
            if len(w1_segment) >= budget:
                w1_segment = w1_segment[:budget]
            response_ids.extend(w1_segment)
            postprocessed_token_segments.append(w1_segment)
            response_length += len(w1_segment)
            idx += 1
            if len(w1_segment) >= budget:
                parallel_length += len(w1_segment)
                break
            budget -= len(w1_segment)

            # Worker 2
            w2_segment = example_token_segments[idx]
            if len(w2_segment) >= budget:
                w2_segment = w2_segment[:budget]
            response_ids.extend(w2_segment)
            postprocessed_token_segments.append(w2_segment)
            response_length += len(w2_segment)
            idx += 1
            if len(w2_segment) >= budget:
                parallel_length += max(len(w1_segment), len(w2_segment))
                break
            budget -= len(w2_segment)

            # Worker 3
            w3_segment = example_token_segments[idx]
            if len(w3_segment) >= budget:
                w3_segment = w3_segment[:budget]
            response_ids.extend(w3_segment)
            postprocessed_token_segments.append(w3_segment)
            response_length += len(w3_segment)
            idx += 1
            if len(w3_segment) >= budget:
                inc = max(len(w1_segment), len(w2_segment))
                inc = max(inc, len(w3_segment))
                parallel_length += inc
                break
            budget -= len(w3_segment)

            # Update parallel length
            parallel_inc = max(len(w1_segment), len(w2_segment))
            parallel_inc = max(parallel_inc, len(w3_segment))
            parallel_length += parallel_inc

            # Update additional parallel metrics
            number_of_parallel_rounds += 1
            current_avg_length = (len(w1_segment) + len(w2_segment) + len(w3_segment)) / 3
            avg_round_lengths.append(current_avg_length)
            current_max_length = parallel_inc
            max_round_lengths.append(current_max_length)
            current_min_length = min(len(w1_segment), len(w2_segment))
            current_min_length = min(current_min_length, len(w3_segment))
            min_round_lengths.append(current_min_length)

        return {
            "non_dup_worker_response_ids": response_ids,
            "response_length": response_length,
            "parallel_length": parallel_length,
            "number_of_parallel_rounds": number_of_parallel_rounds,
            "avg_round_lengths": avg_round_lengths,
            "max_round_lengths": max_round_lengths,
            "min_round_lengths": min_round_lengths,
            "postprocessed_token_segments": postprocessed_token_segments,
        }
    
    # In the following version, there is a limit on the parallel length - however,
    # there is also an additional limit on the total length so that the training sequence
    # will fit into SDPA's limit (so that SDPA does not crash). This is useful if
    # we want to allow a large parallel length, but the DP doesn't need to be
    # excessively high.
    @staticmethod
    def get_non_worker_duplicate_response_tensor_and_metrics_parallel_limit_with_additional_total_limit(parallel_length_budget, total_length_budget, example_token_segments):
        response_ids = []
        response_length = 0
        parallel_length = 0

        # Additional parallel metrics
        number_of_parallel_rounds = 0
        avg_round_lengths = []
        max_round_lengths = []
        min_round_lengths = []

        # postprocessed token segments
        # In case the token segments exceed the budget (either parallel length or total length),
        # we deal with that case here.
        postprocessed_token_segments = []

        idx = 0
        while idx < len(example_token_segments) and parallel_length_budget > 0 and total_length_budget > 0:
            # Single thread segment
            current_segment = example_token_segments[idx]
            if len(current_segment) > parallel_length_budget or len(current_segment) > total_length_budget:
                min_budget = min(parallel_length_budget, total_length_budget)
                current_segment = current_segment[:min_budget]
            response_ids.extend(current_segment)
            postprocessed_token_segments.append(current_segment)
            response_length += len(current_segment)
            parallel_length += len(current_segment)
            parallel_length_budget -= len(current_segment)
            total_length_budget -= len(current_segment)
            idx += 1
            if idx >= len(example_token_segments):
                # Last segment was single thread
                break
            if parallel_length_budget <= 0 or total_length_budget <= 0:
                break

            # Extract worker segments
            w1_segment = example_token_segments[idx]
            idx += 1
            w2_segment = example_token_segments[idx]
            idx += 1
            w3_segment = example_token_segments[idx]
            idx += 1

            # Truncate workers according to parallel length budget
            if len(w1_segment) >= parallel_length_budget:
                w1_segment = w1_segment[:parallel_length_budget]
            if len(w2_segment) >= parallel_length_budget:
                w2_segment = w2_segment[:parallel_length_budget]
            if len(w3_segment) >= parallel_length_budget:
                w3_segment = w3_segment[:parallel_length_budget]
            
            # Truncate workers according to total length budget
            if len(w1_segment) >= total_length_budget:
                w1_segment = w1_segment[:total_length_budget]
                w2_segment = None
                w3_segment = None
            elif len(w1_segment) + len(w2_segment) >= total_length_budget:
                # Since w1 segment is strictly less than total length budget,
                # this means we can truncate w2 segment and it will have at
                # least 1 token
                w2_budget = total_length_budget - len(w1_segment)
                w2_segment = w2_segment[:w2_budget]
                w3_segment = None
            elif len(w1_segment) + len(w2_segment) + len(w3_segment) >= total_length_budget:
                # Now w1 segment and w2 segment together are strictly less
                # than total length budget.
                w3_budget = total_length_budget - len(w1_segment) - len(w2_segment)
                w3_segment = w3_segment[:w3_budget]
            
            # Update response IDs according to segments
            response_ids.extend(w1_segment)
            if w2_segment is not None:
                response_ids.extend(w2_segment)
            if w3_segment is not None:
                response_ids.extend(w3_segment)
            
            # Update postprocessed token segments
            postprocessed_token_segments.append(w1_segment)
            if w2_segment is not None:
                postprocessed_token_segments.append(w2_segment)
            if w3_segment is not None:
                postprocessed_token_segments.append(w3_segment)
            
            # Update response length
            current_round_response_len = len(w1_segment)
            if w2_segment is not None:
                current_round_response_len += len(w2_segment)
            if w3_segment is not None:
                current_round_response_len += len(w3_segment)
            response_length += current_round_response_len

            # Update parallel length
            current_round_parallel_len = len(w1_segment)
            if w2_segment is not None:
                current_round_parallel_len = max(current_round_parallel_len, len(w2_segment))
            if w3_segment is not None:
                current_round_parallel_len = max(current_round_parallel_len, len(w3_segment))
            parallel_length += current_round_parallel_len

            # Update parallel and total length budgets
            parallel_length_budget -= current_round_parallel_len
            total_length_budget -= current_round_response_len

            # Update additional parallel metrics
            number_of_parallel_rounds += 1
            current_avg_length = current_round_response_len / 3
            avg_round_lengths.append(current_avg_length)
            max_round_lengths.append(current_round_parallel_len)
            current_round_min_len = len(w1_segment)
            if w2_segment is not None:
                current_round_min_len = min(current_round_min_len, len(w2_segment))
            else:
                current_round_min_len = 0
            if w3_segment is not None:
                current_round_min_len = min(current_round_min_len, len(w3_segment))
            else:
                current_round_min_len = 0
            min_round_lengths.append(current_round_min_len)

        return {
            "non_dup_worker_response_ids": response_ids,
            "response_length": response_length,
            "parallel_length": parallel_length,
            "number_of_parallel_rounds": number_of_parallel_rounds,
            "avg_round_lengths": avg_round_lengths,
            "max_round_lengths": max_round_lengths,
            "min_round_lengths": min_round_lengths,
            "postprocessed_token_segments": postprocessed_token_segments,
        }

    def get_non_worker_duplicate_response_tensor_and_metrics(self, example_token_segments):
        if self.length_limit_parallel:
            if self.additional_total_length_limit is not None:
                parallel_length_budget = self.config.response_length
                total_length_budget = self.additional_total_length_limit
                print(f"Truncating responses with parallel length budget {parallel_length_budget} and total length budget {total_length_budget}")
                return self.get_non_worker_duplicate_response_tensor_and_metrics_parallel_limit_with_additional_total_limit(
                    parallel_length_budget=parallel_length_budget,
                    total_length_budget=total_length_budget,
                    example_token_segments=example_token_segments,
                )
            else:
                return self.get_non_worker_duplicate_response_tensor_and_metrics_parallel_limit(example_token_segments)
        else:
            return self.get_non_worker_duplicate_response_tensor_and_metrics_total_limit(example_token_segments)

    # Given the token segments returned by process_all_prompts,
    # create new token segments where the worker tokens are there twice,
    # (due to adding a new segment where all workers are together and
    # attended to using a causal mask) and also the corresponding position ids.
    def get_token_segments_and_pos_ids(self, token_segments):
        """
        token_segments: List[List[int]]
        token_segments corresponds to the portions of the response
        for a single example/prompt.
        """

        new_token_segments = []
        new_pos_ids_segments = []

        idx = 0
        pos_id_offset = 0
        while idx < len(token_segments):
            # Single thread segment
            current_segment = token_segments[idx]
            new_token_segments.append(deepcopy(current_segment))
            new_pos_ids = [pos_id_offset + i for i in range(len(current_segment))]
            new_pos_ids_segments.append(new_pos_ids)
            idx += 1
            if idx >= len(token_segments):
                # Last single thread segment
                break
            pos_id_offset = pos_id_offset + len(current_segment)

            # Worker 1
            w1_segment = token_segments[idx]
            new_token_segments.append(deepcopy(w1_segment))
            w1_pos_ids = [pos_id_offset + i for i in range(len(w1_segment))]
            new_pos_ids_segments.append(w1_pos_ids)
            idx += 1
            if idx >= len(token_segments):
                break

            # Worker 2
            w2_segment = token_segments[idx]
            new_token_segments.append(deepcopy(w2_segment))
            w2_pos_ids = [pos_id_offset + i for i in range(len(w2_segment))]
            new_pos_ids_segments.append(w2_pos_ids)
            idx += 1
            if idx >= len(token_segments):
                break

            # Worker 3
            w3_segment = token_segments[idx]
            new_token_segments.append(deepcopy(w3_segment))
            w3_pos_ids = [pos_id_offset + i for i in range(len(w3_segment))]
            new_pos_ids_segments.append(w3_pos_ids)
            idx += 1
            if idx >= len(token_segments):
                break

            # All workers segment
            # Note that this is only there if there is a following
            # single thread segment. If worker 3 is the last segment,
            # then this would not be used anyways.
            all_workers_segment = w1_segment + w2_segment + w3_segment
            new_token_segments.append(all_workers_segment)
            all_workers_pos_ids = [pos_id_offset + i for i in range(len(all_workers_segment))]
            new_pos_ids_segments.append(all_workers_pos_ids)
            pos_id_offset = pos_id_offset + len(all_workers_segment)
        
        return new_token_segments, new_pos_ids_segments

if __name__ == "__main__":
    example_token_segments = [
        [1, 2, 3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12, 13],
        [14, 15, 16],
    ]
    parallel_length_budget = 7
    total_length_budget = 9
    ret_value = vLLMRolloutParallelWorkers.get_non_worker_duplicate_response_tensor_and_metrics_parallel_limit_with_additional_total_limit(
        parallel_length_budget=parallel_length_budget,
        total_length_budget=total_length_budget,
        example_token_segments=example_token_segments,
    )
    print(f"Response IDs: {ret_value['non_dup_worker_response_ids']}")
    print(f"Response length: {ret_value['response_length']}")
    print(f"Parallel length: {ret_value['parallel_length']}")
    print(f"Number of parallel rounds: {ret_value['number_of_parallel_rounds']}")
    print(f"Postprocessed token segments: {ret_value['postprocessed_token_segments']}")