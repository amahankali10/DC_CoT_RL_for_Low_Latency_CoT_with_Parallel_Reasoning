#!/bin/bash
#SBATCH --job-name=parallel_sft
#SBATCH --output=parallel_sft.out
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000123
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=800G
#SBATCH --tasks-per-node=1

eval "$(conda shell.bash hook)"
conda init
conda activate verl
set -a; source ../.env; set +a
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=0

main_output_dir=${MAIN_OUTPUT_DIR}
project_name=${SFT_WANDB_PROJECT}
run_name=${SFT_RUN_NAME}
full_output_dir=$main_output_dir/$project_name/$run_name

train_file_name=${SFT_REWRITE_RESULTS_FILENAME}
train_file_path=${SFT_DATASET_FOLDER}/$train_file_name
eval_file_name=aime_2024_validation.jsonl
eval_file_path=${SFT_DATASET_FOLDER}/$eval_file_name
init_model_name=agentica-org/DeepScaleR-1.5B-Preview

srun accelerate launch --main_process_port=1234 --config_file "accelerate_configs/deepseek_r1_distill_8_GPUs.yaml" new_generation_sft_script.py \
    --model_name $init_model_name \
    --resume_from_hf_checkpoint False \
    --train_file_path $train_file_path \
    --eval_file_path $eval_file_path \
    --rollout_data_dir $full_output_dir/validation_rollout_data \
    --use_AIME_format_val True \
    --use_parallel_prompt True \
    --wandb_project $project_name \
    --output_dir $full_output_dir \
    --eval_strategy "no" \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-6 \
    --optim "paged_adamw_32bit" \
    --num_train_epochs 5 \
    --lr_scheduler_type "warmup_stable_decay" \
    --wsd_decay_steps 100 \
    --wsd_decay_type "linear" \
    --warmup_ratio 0.05 \
    --logging_strategy "steps" \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 200 \
    --bf16 True \
    --tf32 True \
    --run_name $run_name \
    --eval_best_attempts 20 \
    --train_input_ids_length_limit 10000 \
    --eval_step_interval 200 \
    --eval_epoch_end True \
    --eval_before_training True \
    --ddp_timeout 7200 \
    --vllm_dtype "bfloat16" \
    --vllm_block_size 64 \
    --vllm_seed 42 \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_max_num_batched_tokens 40000 \
    --vllm_max_num_seqs 2000 \
    --vllm_enforce_eager True \
    --vllm_max_model_len 40000 \
    --vllm_max_seq_len_to_capture 40000 \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_new_tokens 24000 \
    --parallel_length_upper_bound 40000 \
    --max_parallel_rounds 20