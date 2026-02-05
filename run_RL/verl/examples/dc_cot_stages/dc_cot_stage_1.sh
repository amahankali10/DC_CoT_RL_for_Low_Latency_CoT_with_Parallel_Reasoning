#!/bin/bash
#SBATCH --job-name=dc_cot_stage_1
#SBATCH --output=dc_cot_stage_1.out
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH -A marlowe-m000123-pm04
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=800G
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00

eval "$(conda shell.bash hook)"
conda activate verl
set -a; source ../../.env; set +a
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_CONFIGURE_LOGGING=0
export VLLM_USE_V1=0
export VLLM_USE_V1_MULTIPROCESSING=0
export RAY_ADDRESS="local"

# Per-job Ray temp directory on node-local storage to avoid shared path clashes
export RAY_TMPDIR=/tmp/$USER/ray/${SLURM_JOB_ID}
# Create on head now; on workers we create inside their srun command
mkdir -p "$RAY_TMPDIR" || true

echo "=== GPU Diagnostics ==="
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_PROCID: $SLURM_PROCID"

python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"No CUDA\"}')"

main_output_dir=${MAIN_OUTPUT_DIR}

project_name=${RL_WANDB_PROJECT}
run_name=${RL_STAGE1_NAME}

pw_sft_project=${SFT_WANDB_PROJECT}
pw_sft_run_name=${SFT_RUN_NAME}
init_model_path=$main_output_dir/$pw_sft_project/$pw_sft_run_name/checkpoint-1800

train_test_folder=${RL_PARQUET_DATASETS}/aime_2024_parallel_prompt_with_special_tokens

unset ROCR_VISIBLE_DEVICES

python3 -u -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_test_folder/train_deepscaler.parquet \
    data.val_files=$train_test_folder/test_aime.parquet \
    data.train_batch_size=96 \
    data.gen_batch_size=288 \
    data.max_prompt_length=1024 \
    data.max_response_length=7500 \
    +data.seed=100 \
    actor_rollout_ref.model.path=$init_model_path \
    actor_rollout_ref.actor.optim.lr=7.071e-7 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=48 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_parallel_worker \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_k=10000 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=7 \
    actor_rollout_ref.rollout.val_kwargs.n=50 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 \
    +actor_rollout_ref.nccl_timeout=36000 \
    +actor_rollout_ref.model.attn_implementation=sdpa \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.remove_chat_template=True \
    +actor_rollout_ref.model.shard_grad_op_only=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref.actor.bsz_total_tokens_log_prob=100000 \
    +actor_rollout_ref.actor.bsz_total_examples_log_prob=1 \
    +actor_rollout_ref.actor.bsz_bscs_product_log_prob=6000 \
    +actor_rollout_ref.actor.bsz_total_tokens_update=100000 \
    +actor_rollout_ref.actor.bsz_total_examples_update=1 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean-grad-acc-1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.load_format=dummy_dtensor \
    actor_rollout_ref.rollout.max_num_batched_tokens=200000 \
    +actor_rollout_ref.rollout.block_size=64 \
    actor_rollout_ref.rollout.max_num_seqs=2000 \
    +actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    +actor_rollout_ref.rollout.length_limit_parallel=True \
    +actor_rollout_ref.rollout.max_parallel_rounds=1000 \
    +actor_rollout_ref.rollout.max_tokens_in_parallel_round=40000 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref.ref.bsz_total_tokens_log_prob=100000 \
    +actor_rollout_ref.ref.bsz_total_examples_log_prob=1 \
    +actor_rollout_ref.ref.bsz_bscs_product_log_prob=6000 \
    reward_model.reward_manager=dapo_parallel_worker \
    +reward_model.use_deepscaler_reward=False \
    +reward_model.correct_and_no_format_reward=0.5 \
    +reward_model.format_and_incorrect_reward=0.0 \
    +reward_model.num_examine_train=4 \
    +reward_model.num_examine_val=4 \
    reward_model.enable=False \
    trainer.rollout_data_dir=$main_output_dir/$project_name/$run_name/rollout_data \
    +trainer.validation_data_dir=$main_output_dir/$project_name/$run_name/validation_rollout_data \
    trainer.default_local_dir=$main_output_dir/$project_name/$run_name \
    trainer.resume_mode=auto \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=-1 \
    algorithm.filter_groups.metric=all_wrong_or_no_format_filter_metric \
    reward_model.overlong_buffer.enable=True \
    +reward_model.overlong_buffer.parallel=True \
    reward_model.overlong_buffer.len=5500 \
    reward_model.overlong_buffer.penalty_factor=0.1 \
    reward_model.overlong_buffer.log=True \
    +reward_model.overlong_buffer.penalize_only_correct_resps=True \
    +reward_model.enforce_answer_between_answer_tags=False \
    +reward_model.prepend_think_token_response=True