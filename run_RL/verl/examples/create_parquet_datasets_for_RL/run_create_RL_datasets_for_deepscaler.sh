#!/usr/bin/env bash
#SBATCH --job-name=dscl_repro_dataset
#SBATCH --nodes=1
#SBATCH --partition=tiger
#SBATCH --nodelist=tiger8
#SBATCH --account=tiger
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --output=dscl_repro_dataset.out

eval "$(conda shell.bash hook)"
conda activate verl
set -a; source ../../../../.env; set +a
export PYTHONUNBUFFERED=1

python create_deepscaler_repro_dataset_correct_prompt_template.py --val_dataset_path HuggingFaceH4/aime_2024 --local_dir ${RL_PARQUET_DATASETS}/aime_2024_standard_prompt_with_special_tokens

python create_deepscaler_repro_dataset_correct_prompt_template.py --val_dataset_path HuggingFaceH4/MATH-500 --local_dir ${RL_PARQUET_DATASETS}/math_500_standard_prompt_with_special_tokens

python create_deepscaler_repro_dataset_correct_prompt_template.py --val_dataset_path math-ai/amc23 --local_dir ${RL_PARQUET_DATASETS}/amc_23_standard_prompt_with_special_tokens

python create_deepscaler_repro_dataset_correct_prompt_template.py --val_dataset_path math-ai/minervamath --local_dir ${RL_PARQUET_DATASETS}/minerva_math_standard_prompt_with_special_tokens

python create_deepscaler_repro_dataset_correct_prompt_template.py --val_dataset_path MathArena/hmmt_feb_2025 --local_dir ${RL_PARQUET_DATASETS}/hmmt_feb_2025_standard_prompt_with_special_tokens

python create_deepscaler_repro_dataset_correct_prompt_template.py --val_dataset_path Hothan/OlympiadBench --local_dir ${RL_PARQUET_DATASETS}/olympiad_bench_standard_deepscaler_prompt_with_special_tokens
