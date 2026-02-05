"""
Based on sft.py from s1 repo
https://github.com/simplescaling/s1/blob/main/train/sft.py

We will use this script to either train the Qwen base model or the
distilled Deepseek R1 model. Thus, we will use the Deepseek R1 prompt
rather than the chat template.
"""
from prepare_data_for_new_parallel_sft import ParallelWorkersSFTDataset, DataCollatorForParallelWorkerSFT, deepseek_r1_prompt, deepseek_r1_prompt_parallel
# from vllm_generation_callback_for_sft import ParallelGenerationEvalCallback
from vllm_generation_callback_for_sft_update_weights_dynamic import ParallelGenerationEvalCallback
from override_qwen2_attn_mask import Qwen2ForCausalLMOverride, Qwen2ModelOverride

import logging
import os
import warnings
import jsonlines
from types import SimpleNamespace, FunctionType, MethodType

from typing import Optional, Dict, Union
from dataclasses import dataclass, field, asdict

import transformers
import torch.nn as nn

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)
    # Enable model parallelism
    labels = labels.to(logits.device)
    loss = fixed_cross_entropy(logits, labels, num_items_in_batch, ignore_index, **kwargs)
    return loss

@dataclass
class TrainingConfig:
    # Basic training parameters
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    resume_from_hf_checkpoint: bool = field(default=False)
    wandb_project: Optional[str] = field(default="parallel_workers_initial_sft_Big_MATH")
    wandb_entity: Optional[str] = field(default="parallel-workers")
    train_file_path: Optional[str] = field(default=None)
    eval_file_path: Optional[str] = field(default=None)
    eval_best_attempts: Optional[int] = field(default=5)
    max_train_examples: Optional[int] = field(default=None)
    max_val_examples: Optional[int] = field(default=None)
    train_input_ids_length_limit: Optional[int] = field(default=None)
    eval_step_interval: Optional[int] = field(default=100)
    eval_epoch_end: Optional[bool] = field(default=True)
    eval_before_training: Optional[bool] = field(default=True)
    use_parallel_prompt: Optional[bool] = field(default=False)
    use_AIME_format_val: Optional[bool] = field(default=True)
    rollout_data_dir: Optional[str] = field(default=None)

    # vLLM Engine Arguments
    vllm_dtype: str = field(default="bfloat16")
    vllm_block_size: int = field(default=64)
    vllm_seed: int = field(default=42)
    vllm_gpu_memory_utilization: float = field(default=0.95)
    vllm_max_num_batched_tokens: int = field(default=30000)
    vllm_max_num_seqs: int = field(default=2000)
    vllm_enforce_eager: bool = field(default=True)
    vllm_max_model_len: int = field(default=16384)
    vllm_max_seq_len_to_capture: int = field(default=16384)

    # Generation Arguments
    temperature: float = field(default=0.6)  # 0 for greedy
    top_p: float = field(default=0.95)
    max_new_tokens: int = field(default=16384)
    parallel_length_upper_bound: int = field(default=1000)
    max_parallel_rounds: int = field(default=20)

    # WSD arguments
    wsd_decay_steps: int = field(default=-1)
    wsd_decay_type: str = field(default="linear")

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity

    def get_vllm_engine_args(self) -> Union[Dict, SimpleNamespace]:
        """Get vLLM engine arguments as a SimpleNamespace object that supports both dict and attribute access."""
        args_dict = {
            "dtype": self.vllm_dtype,
            "block_size": self.vllm_block_size,
            "seed": self.vllm_seed,
            "gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "max_num_batched_tokens": self.vllm_max_num_batched_tokens,
            "max_num_seqs": self.vllm_max_num_seqs,
            "enforce_eager": self.vllm_enforce_eager,
            "max_model_len": self.vllm_max_model_len,
            "max_seq_len_to_capture": self.vllm_max_seq_len_to_capture,
        }
        return SimpleNamespace(**args_dict)

    def get_generation_args(self) -> Union[Dict, SimpleNamespace]:
        """Get generation arguments as a SimpleNamespace object that supports both dict and attribute access."""
        args_dict = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "parallel_length_upper_bound": self.parallel_length_upper_bound,
            "max_parallel_rounds": self.max_parallel_rounds,
            "top_p": self.top_p,
        }
        return SimpleNamespace(**args_dict)

def extract_ground_truth_answer(completion):
    from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
    string_in_last_boxed = last_boxed_only_string(completion)
    assert string_in_last_boxed is not None
    answer = remove_boxed(string_in_last_boxed)
    return answer

def postprocess_eval_data(eval_data, use_parallel_prompt, use_AIME_format):
    """
    Format the prompts and extract ground truth answers
    """
    prompt_ground_truth_pairs = []

    if use_parallel_prompt:
        prompt_template = deepseek_r1_prompt_parallel
    else:
        prompt_template = deepseek_r1_prompt

    for item in eval_data:
        problem = item["problem"]
        prompt = prompt_template.format(question=problem)
        if use_AIME_format:
            ground_truth_answer = item["answer"]
        else:
            solution = item["solution"]
            ground_truth_answer = extract_ground_truth_answer(solution)
        example_dict = {"prompt": prompt, "ground_truth_answer": ground_truth_answer}
        prompt_ground_truth_pairs.append(example_dict)
    return prompt_ground_truth_pairs

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, transformers.TrainingArguments))
    config, args = parser.parse_args_into_dataclasses()
    if args.lr_scheduler_type == "warmup_stable_decay":
        assert config.wsd_decay_steps != -1
        args.lr_scheduler_kwargs = {
            "num_decay_steps": config.wsd_decay_steps,
            "decay_type": config.wsd_decay_type
        }

    args.remove_unused_columns = False
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {}
    kwargs["attn_implementation"] = "sdpa"
    if "70B" in config.model_name:
        raise NotImplementedError
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto", "use_cache": False}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
        setattr(model, "forward", MethodType(Qwen2ForCausalLMOverride.forward, model))
        setattr(model.model, "_update_causal_mask", MethodType(Qwen2ModelOverride._update_causal_mask, model.model))
    
    model.loss_function = ForCausalLMLoss
    
    # Load training data from JSONLines file
    print(f"Loading training data from {config.train_file_path}")
    train_data = []
    with jsonlines.open(config.train_file_path, mode='r') as reader:
        for item in reader:
            train_data.append(item)
    print(f"Loaded {len(train_data)} examples from training file")
    if config.max_train_examples is not None:
        train_data = train_data[:config.max_train_examples]
        print(f"Truncated training data to {len(train_data)} examples")

    # Load eval data from JSONLines file
    print(f"Loading eval data from {config.eval_file_path}")
    eval_examples = []
    with jsonlines.open(config.eval_file_path, mode="r") as reader:
        for item in reader:
            eval_examples.append(item)
    eval_examples = postprocess_eval_data(eval_examples, config.use_parallel_prompt, config.use_AIME_format_val)
    print(f"Loaded {len(eval_examples)} examples from eval file")
    if config.max_val_examples is not None:
        eval_examples = eval_examples[:config.max_val_examples]
        print(f"Truncated eval data to {len(eval_examples)} examples")

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    qwen_base_models = []
    qwen_base_models.append("Qwen/Qwen2.5-7B")
    qwen_base_models.append("Qwen/Qwen2.5-3B")
    qwen_base_models.append("Qwen/Qwen2.5-1.5B")
    qwen_deepseek_distilled_models = []
    qwen_deepseek_distilled_models.append("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    qwen_deepseek_distilled_models.append("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    # if config.model_name in qwen_base_models or config.model_name in qwen_deepseek_distilled_models:
    tokenizer.pad_token = "<|fim_pad|>"
    tokenizer.pad_token_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
    # else:
    #     raise ValueError(f"Model {config.model_name} not supported yet")
    
    dataset = ParallelWorkersSFTDataset(train_data, tokenizer, config.use_parallel_prompt)
    if config.train_input_ids_length_limit is not None:
        dataset.remove_long_examples(config.train_input_ids_length_limit)
        print(f"Removed {len(train_data) - len(dataset)} examples from training data")
    collator = DataCollatorForParallelWorkerSFT(tokenizer)

    trainer = transformers.Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )
    eval_callback = ParallelGenerationEvalCallback(
        trainer=trainer,
        eval_dataset=eval_examples,
        tokenizer=tokenizer,
        generation_args=config.get_generation_args(),
        vllm_engine_args=config.get_vllm_engine_args(),
        eval_step_interval=config.eval_step_interval,
        eval_epoch_end=config.eval_epoch_end,
        eval_best_attempts=config.eval_best_attempts,
        eval_before_training=config.eval_before_training,
        base_model_path=config.model_name,
        rollout_data_dir=config.rollout_data_dir
    )
    trainer.add_callback(eval_callback)

    trainer.train(resume_from_checkpoint=config.resume_from_hf_checkpoint)
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()    

if __name__ == "__main__":
    train()