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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.utils.dataset.rl_dataset import collate_fn

from .dapo_ray_trainer import RayDAPOTrainer


@hydra.main(config_path="config", config_name="dapo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # connect to external cluster if provided, otherwise auto-detect
        ray.init(
            address=os.environ.get("RAY_ADDRESS", "auto"),
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    if OmegaConf.select(config.trainer, "profile_steps") is not None and len(OmegaConf.select(config.trainer, "profile_steps")) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy == "fsdp":
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        from verl.workers.reward_manager import get_reward_manager_cls

        # Note(haibin.lin): please make sure custom reward managers are imported and
        # registered via `verl.workers.reward_manager.register`
        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        reward_manager_cls = get_reward_manager_cls(reward_manager_name)

        compute_score = get_custom_reward_fn(config)
        if config.actor_rollout_ref.rollout.name == "vllm_parallel_worker":
            assert reward_manager_name == "dapo_parallel_worker"
            reward_fn = reward_manager_cls(
                tokenizer=tokenizer,
                num_examine=config.reward_model.get("num_examine_train", 1),
                compute_score=compute_score,
                length_penalty_coeff=config.reward_model.get("parallel_worker_length_penalty_coeff", 0.0),
                correct_and_no_format_reward=config.reward_model.get("correct_and_no_format_reward", None),
                format_and_incorrect_reward=config.reward_model.get("format_and_incorrect_reward", None),
                reward_fn_key=config.data.reward_fn_key,
                max_resp_len=config.data.max_response_length,
                overlong_buffer_cfg=config.reward_model.overlong_buffer,
                total_tokens_incentive=config.reward_model.get("total_tokens_incentive", 0.0),
                use_deepscaler_reward=config.reward_model.get("use_deepscaler_reward", False),
                enforce_answer_between_answer_tags=config.reward_model.get("enforce_answer_between_answer_tags", False),
                prepend_think_token_response=config.reward_model.get("prepend_think_token_response", False),
            )
            val_reward_fn = reward_manager_cls(
                tokenizer=tokenizer,
                num_examine=config.reward_model.get("num_examine_val", 1),
                compute_score=compute_score,
                length_penalty_coeff=config.reward_model.get("parallel_worker_length_penalty_coeff", 0.0),
                correct_and_no_format_reward=config.reward_model.get("correct_and_no_format_reward", None),
                format_and_incorrect_reward=config.reward_model.get("format_and_incorrect_reward", None),
                reward_fn_key=config.data.reward_fn_key,
                max_resp_len=config.data.max_response_length,
                overlong_buffer_cfg=config.reward_model.overlong_buffer,
                total_tokens_incentive=config.reward_model.get("total_tokens_incentive", 0.0),
                use_deepscaler_reward=config.reward_model.get("use_deepscaler_reward", False),
                enforce_answer_between_answer_tags=config.reward_model.get("enforce_answer_between_answer_tags", False),
                prepend_think_token_response=config.reward_model.get("prepend_think_token_response", False),
            )
        else:
            reward_fn = reward_manager_cls(
                tokenizer=tokenizer,
                num_examine=0,
                compute_score=compute_score,
                reward_fn_key=config.data.reward_fn_key,
                max_resp_len=config.data.max_response_length,
                overlong_buffer_cfg=config.reward_model.overlong_buffer,
            )

            # Note that we always use function-based RM for validation
            val_reward_fn = reward_manager_cls(
                tokenizer=tokenizer,
                num_examine=1,
                compute_score=compute_score,
                reward_fn_key=config.data.reward_fn_key,
                max_resp_len=config.data.max_response_length,
                overlong_buffer_cfg=config.reward_model.overlong_buffer,
            )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        if config.data.get("remove_chat_template", None):
            # Tokenizer apply_chat_template does nothing
            # This is correct because the special tokens are
            # already included in the parquet file.
            tokenizer.apply_chat_template = lambda messages, add_generation_prompt, tokenize=False: tokenizer.encode(messages[0]["content"]) if tokenize else messages[0]["content"]

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_val=False)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_val=True)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
