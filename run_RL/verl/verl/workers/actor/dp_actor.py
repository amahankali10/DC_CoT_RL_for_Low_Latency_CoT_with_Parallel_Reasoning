# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import itertools
import logging
import os
import json
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import gc

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.actor.reconstruct_attn_mask_parallel import get_full_parallel_attn_mask

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

from tqdm import tqdm

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None, tokenizer=None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

        self.tokenizer = tokenizer

        # Parallel workers dynamic batch size configurations
        self.bsz_total_tokens_log_prob = self.config.get("bsz_total_tokens_log_prob", 60000)
        self.bsz_total_examples_log_prob = self.config.get("bsz_total_examples_log_prob", 12)
        self.bsz_bscs_product_log_prob = self.config.get("bsz_bscs_product_log_prob", 8000)
        # 
        self.bsz_total_tokens_update = self.config.get("bsz_total_tokens_update", 100000)
        self.bsz_total_examples_update = self.config.get("bsz_total_examples_update", 1)

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, parallel_workers=False, bsz_bscs_product=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            parallel_workers: bool
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            if parallel_workers:
                # Reconstruct the 4D attention mask
                token_segment_lengths = micro_batch["token_segment_lengths"]
                prompt_padding_mask = micro_batch["prompt_padding_mask"]
                max_input_id_len = input_ids.shape[1] - prompt_padding_mask.shape[1]
                attention_mask = get_full_parallel_attn_mask(
                    prompt_padding_mask=prompt_padding_mask,
                    segment_lengths=token_segment_lengths,
                    max_input_id_len=max_input_id_len,
                )
                assert not self.use_remove_padding, "Remove padding is not supported for parallel workers due to custom mask/position_ids."
            else:
                attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp

                ########### Remove excessive padding ###########
                padding_mask = input_ids != self.tokenizer.pad_token_id
                bs, seq_len = padding_mask.size()
                global_max = None
                global_min = None
                for i in range(bs):
                    nnz = torch.nonzero(padding_mask[i])
                    nnz_max = torch.max(nnz).item()
                    nnz_min = torch.min(nnz).item()
                    if global_max is None or nnz_max > global_max:
                        global_max = nnz_max
                    if global_min is None or nnz_min < global_min:
                        global_min = nnz_min
                num_left_padding = global_min
                num_right_padding = seq_len - (global_max + 1)
                input_ids = input_ids[:, global_min:global_max + 1]
                if attention_mask.dim() == 4:
                    attention_mask = attention_mask[:, :, global_min:global_max + 1, global_min:global_max + 1]
                else:
                    assert attention_mask.dim() == 2
                    raise NotImplementedError
                position_ids = position_ids[:, global_min:global_max + 1]
                ###########

                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    raise NotImplementedError("Did not yet implement remove padding with fused kernels")
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits
                    logits.div_(temperature)

                    if num_right_padding > 0:
                        response_ids = micro_batch["responses"][:, :-num_right_padding]
                    else:
                        response_ids = micro_batch["responses"]
                    _, actual_response_length = response_ids.shape
                    logits = logits[:, -actual_response_length - 1 : -1, :]  # (bsz, actual_response_length, vocab_size)
                    BSCS_PRODUCT = bsz_bscs_product
                    bs, _, _ = logits.shape
                    log_probs = logprobs_from_logits(logits, response_ids)
                    if calculate_entropy:
                        chunk_size = BSCS_PRODUCT // bs
                        entropy = verl_F.entropy_from_logits_with_chunking(logits, chunk_size=chunk_size)  # (bsz, actual_response_length)
                    
                    # Pad the log probs and entropy - only add right padding
                    if num_right_padding > 0:
                        right_pad_tensor = torch.zeros((bs, num_right_padding), device=log_probs.device, dtype=log_probs.dtype)
                        log_probs = torch.cat([log_probs, right_pad_tensor], dim=1)
                        if calculate_entropy:
                            entropy = torch.cat([entropy, right_pad_tensor], dim=1)

                    ########### Add back padding to logits ###########
                    # left_pad_tensor = torch.zeros((bs, num_left_padding, logits.size(-1)), device=logits.device, dtype=logits.dtype)
                    # right_pad_tensor = torch.zeros((bs, num_right_padding, logits.size(-1)), device=logits.device, dtype=logits.dtype)
                    # logits = torch.cat([left_pad_tensor, logits, right_pad_tensor], dim=1)
                    ###########
                    
                    if parallel_workers:
                        hardcoded_token_masks = micro_batch["hardcoded_token_masks"]
                        log_probs = log_probs * hardcoded_token_masks
                        if calculate_entropy:
                            entropy = entropy * hardcoded_token_masks

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False, parallel_workers=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

                ``parallel_attn_mask_4D``: tensor of shape [batch_size, 1, prompt_length + response_length, prompt_length + response_length]. torch.bool.
                - present if parallel_workers is True

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        gc.collect()
        torch.cuda.empty_cache()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if parallel_workers:
            select_keys.append("token_segment_lengths")
            select_keys.append("prompt_padding_mask")
            select_keys.append("hardcoded_token_masks")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)
        
        # Sort batches by number of non-padding tokens
        # Then, make as big a batch as possible
        TOTAL_TOKENS_LIMIT = self.bsz_total_tokens_log_prob
        TOTAL_EXAMPLES_LIMIT = self.bsz_total_examples_log_prob
        if parallel_workers:
            assert micro_batch_size == 1 and not use_dynamic_bsz and not has_multi_modal_inputs

            # 1. Sort by number of non-padding tokens
            token_counts_with_idx = []
            for idx, mb in enumerate(micro_batches):
                ids = mb["input_ids"]
                nonpad = (ids != self.tokenizer.pad_token_id).sum().item()
                token_counts_with_idx.append((idx, nonpad))
            token_counts_with_idx.sort(key=lambda x: x[1])
            parallel_sorted_indices = [i for i, _ in token_counts_with_idx]
            parallel_sorted_token_counts = [count for _, count in token_counts_with_idx]
            micro_batches = [micro_batches[i] for i in parallel_sorted_indices]

            # 2. Group into batches of at most TOTAL_TOKENS_LIMIT tokens
            # and at most TOTAL_EXAMPLES_LIMIT examples
            new_micro_batches = []
            running_batch = []
            running_token_count = 0
            print(f"Parallel sorted token counts: {parallel_sorted_token_counts}")
            for mb, count in zip(micro_batches, parallel_sorted_token_counts):
                if running_token_count + count > TOTAL_TOKENS_LIMIT or len(running_batch) + 1 > TOTAL_EXAMPLES_LIMIT:
                    running_batch = torch.cat(running_batch, dim=0)
                    new_micro_batches.append(running_batch)
                    running_batch = []
                    running_token_count = 0
                
                running_batch.append(mb)
                running_token_count += count
            
            if len(running_batch) > 0:
                running_batch = torch.cat(running_batch, dim=0)
                new_micro_batches.append(running_batch)
            
            micro_batches = new_micro_batches

        log_probs_lst = []
        entropy_lst = []
        pbar = tqdm(total=len(micro_batches), desc="Micro batches for computing log probs", leave=True)
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy, parallel_workers=parallel_workers, bsz_bscs_product=self.bsz_bscs_product_log_prob)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            
            pbar.update(1)
        
        pbar.close()

        # Unsort lists back to original order and divide log_probs_lst and entropy_lst into lists of batch size 1
        if parallel_workers:
            new_log_probs_lst = []
            for log_probs in log_probs_lst:
                current_bs, _ = log_probs.shape
                new_log_probs_lst.extend([log_probs[i:i + 1] for i in range(current_bs)])
            log_probs_lst = new_log_probs_lst

            if calculate_entropy:
                new_entropy_lst = []
                for entropy in entropy_lst:
                    current_bs, _ = entropy.shape
                    new_entropy_lst.extend([entropy[i:i + 1] for i in range(current_bs)])
                entropy_lst = new_entropy_lst
            
            revert_indices = get_reverse_idx(parallel_sorted_indices)
            log_probs_lst = [log_probs_lst[i] for i in revert_indices]
            if calculate_entropy:
                entropy_lst = [entropy_lst[i] for i in revert_indices]

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        return log_probs, entropys
    
    def calculate_total_tokens(self, data, parallel_workers, multi_turn):
        responses = data["responses"]
        response_length = responses.size(1)
        if parallel_workers:
            response_mask = data["hardcoded_token_masks"]
        elif multi_turn:
            response_mask = data["loss_mask"][:, -response_length:]
        else:
            attention_mask = data["attention_mask"]
            response_mask = attention_mask[:, -response_length:]
        
        total_tokens = response_mask.sum().item()
        return total_tokens

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto, parallel_workers: bool):
        # make sure we are in training mode
        self.actor_module.train()

        gc.collect()
        torch.cuda.empty_cache()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        train_global_step = data.meta_info["global_steps_for_actor"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if parallel_workers:
            select_keys.append("token_segment_lengths")
            select_keys.append("prompt_padding_mask")
            select_keys.append("hardcoded_token_masks")
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        logging_probability_ratio_lines = []
        for epoch in range(self.config.ppo_epochs):
            pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch} of {self.config.ppo_epochs} epochs of PPO training", leave=True)
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                
                num_tokens_mini_batch = self.calculate_total_tokens(
                    data=mini_batch,
                    parallel_workers=parallel_workers,
                    multi_turn=multi_turn,
                )
                print(f"Num tokens in mini batch: {num_tokens_mini_batch}")

                self.actor_optimizer.zero_grad()

                if parallel_workers:
                    TOTAL_TOKENS_LIMIT = self.bsz_total_tokens_update
                    TOTAL_EXAMPLES_LIMIT = self.bsz_total_examples_update
                    assert self.config.ppo_micro_batch_size_per_gpu == 1
                    assert not self.config.use_dynamic_bsz
                    assert not has_multi_modal_inputs

                    # 1. Sort the micro batches
                    token_counts_with_idx = []
                    for idx, mb in enumerate(micro_batches):
                        ids = mb["input_ids"]
                        nonpad = (ids != self.tokenizer.pad_token_id).sum().item()
                        token_counts_with_idx.append((idx, nonpad))
                    token_counts_with_idx.sort(key=lambda x: x[1])
                    parallel_sorted_indices = [i for i, _ in token_counts_with_idx]
                    parallel_sorted_token_counts = [count for _, count in token_counts_with_idx]
                    micro_batches = [micro_batches[i] for i in parallel_sorted_indices]

                    # 2. Group into larger micro batches
                    new_micro_batches = []
                    running_batch = []
                    running_token_count = 0
                    print(f"Parallel sorted token counts: {parallel_sorted_token_counts}")
                    for mb, count in zip(micro_batches, parallel_sorted_token_counts):
                        if running_token_count + count > TOTAL_TOKENS_LIMIT or len(running_batch) + 1 > TOTAL_EXAMPLES_LIMIT:
                            running_batch = torch.cat(running_batch, dim=0)
                            new_micro_batches.append(running_batch)
                            running_batch = []
                            running_token_count = 0
                        
                        running_batch.append(mb)
                        running_token_count += count
                    
                    if len(running_batch) > 0:
                        running_batch = torch.cat(running_batch, dim=0)
                        new_micro_batches.append(running_batch)
                    
                    micro_batches = new_micro_batches

                print(f"Processing {len(micro_batches)} micro batches")
                for mb_num, data in enumerate(micro_batches):
                    if mb_num % 5 == 0:
                        print(f"Processing micro batch {mb_num} of {len(micro_batches)}")

                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_device_id())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if parallel_workers:
                        response_mask = data["hardcoded_token_masks"]
                    elif multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy, parallel_workers=parallel_workers)

                    print(f"RL loss type: {self.config.rl_loss_type}")
                    print(f"clip ratio high: {clip_ratio_high}")
                    print(f"clip ratio low: {clip_ratio_low}")
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, probability_ratio_logging_dict = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_type=self.config.rl_loss_type,
                        loss_agg_mode=loss_agg_mode,
                    )

                    # Add extra info to the dict
                    probability_ratio_logging_dict["global_rl_step"] = train_global_step
                    probability_ratio_logging_dict["ppo_epoch"] = epoch
                    probability_ratio_logging_dict["step_within_ppo_epoch"] = batch_idx
                    probability_ratio_logging_dict["input_ids_without_shift"] = data["input_ids"].cpu().tolist()
                    logging_probability_ratio_lines.append(probability_ratio_logging_dict)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef
                    
                    assert loss_agg_mode == "token-mean-grad-acc-1"
                    if loss_agg_mode == "token-mean-grad-acc-1":
                        logger.warning_once("Using token-mean-grad-acc-1 loss aggregation mode")
                        loss = policy_loss / num_tokens_mini_batch
                    elif parallel_workers:
                        assert False
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    elif self.config.use_dynamic_bsz:
                        assert False
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        assert False
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }
                    if pg_clipfrac is not None:
                        data["actor/pg_clipfrac"] = pg_clipfrac.detach().item()
                    if pg_clipfrac_lower is not None:
                        data["actor/pg_clipfrac_lower"] = pg_clipfrac_lower.detach().item()

                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                pbar.update(1)
                append_to_dict(metrics, data)
            pbar.close()
        self.actor_optimizer.zero_grad()

        # Log all of the prob ratio dicts to JSONL file
        dump_path = self.config.logging_prob_ratios_path
        os.makedirs(dump_path, exist_ok=True)
        import uuid
        filename = f"{train_global_step}_{uuid.uuid4()}.jsonl"
        filename = os.path.join(dump_path, filename)
        final_lines = []
        for entry in logging_probability_ratio_lines:
            final_lines.append(json.dumps(entry, ensure_ascii=False))
        with open(filename, "w") as f:
            f.write("\n".join(final_lines) + "\n")
        print(f"Dumped update_policy statistics to {filename}")

        return metrics
