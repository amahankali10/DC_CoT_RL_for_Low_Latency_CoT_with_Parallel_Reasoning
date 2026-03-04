# Divide-and-Conquer CoT: RL for Reducing Latency via Parallel Reasoning

This is the official implementation of Divide-and-Conquer CoT (DC-CoT), available at https://arxiv.org/abs/2601.23027.

## Contents
- [Setup - Conda Environment and .env File](#setup---conda-environment-and-env-file)
- [SFT Dataset](#sft-dataset)
  - [Generating Sequential Rollouts](#generating-sequential-rollouts)
  - [Rewriting Sequential Rollouts with Claude Sonnet 4.5](#rewriting-sequential-rollouts-with-claude-sonnet-45)
- [SFT](#sft)
  - [SFT Validation Dataset](#sft-validation-dataset)
  - [SFT Script](#sft-script)
- [RL](#rl)
  - [Creating Training/Validation Datasets in .parquet Format](#creating-trainingvalidation-datasets-in-parquet-format)
  - [DC-CoT - Stages 1 through 4](#dc-cot---stages-1-through-4)
  - [HLP Stage - DC-CoT-HLP and DSR-HLP-12K/DSR-HLP-24K](#hlp-stage---dc-cot-hlp-and-dsr-hlp-12kdsr-hlp-24k)
- [Compute Requirements](#compute-requirements)
- [Citation](#citation)
- [License](#license)

## Models and SFT Dataset

Our models trained with RL can be downloaded at the following links:
- [DC-CoT](https://huggingface.co/amahanka/DC-CoT)
- [DC-CoT-HLP](https://huggingface.co/amahanka/DC-CoT-HLP)

Our SFT dataset can be downloaded [here](https://huggingface.co/datasets/amahanka/DC-CoT-SFT-Dataset).

## Setup - Conda Environment and .env File

The Conda environment can be created as follows:
```
conda env create -f verl_env.yml -n verl
conda activate verl
pip3 install vllm==0.8.4
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
cd run_RL/verl
pip3 install -e . --no-deps
```
We use the verl environment for all of our scripts.

Alternatively, we provide Conda environments in zipped format as follows:
- verl: [Download from Google Drive](https://drive.google.com/file/d/1zm9UYtcTUvOdyHqyFDilykqKeKHf-sjO/view?usp=sharing)
- parallel_sft_2: [Download from Google Drive](https://drive.google.com/file/d/1o1fQe2ImcTXe_yzj39dpxhaK3VZp-JzM/view?usp=sharing)

parallel_sft_2 can be used for our SFT run, while verl is used for all other scripts. These Conda environments were 
compressed using conda pack. Please follow the instructions on the [Conda-Pack webpage](https://conda.github.io/conda-pack/) 
to extract the environments to your machine, to a desirable directory.

Additionally, the scripts below require certain environment variables to be set in .env. We give an example in .env.example,
which shows all the variables that need to be set for all the scripts.

## SFT Dataset

### Generating Sequential Rollouts

We first generate a dataset of sequential CoTs with [DeepScaleR-1.5B-Preview](agentica-org/DeepScaleR-1.5B-Preview), 
using the following commands (run from the main directory):
```
set -a; source .env; set +a
cd create_SFT_dataset_files
python generate_deepscaler_model_dataset_responses_UPDATED.py --subsample-start 0 --subsample-end 15000
```

Before doing so, please set the output folder in .env, as SFT_DATASET_FOLDER. Then, the sequential CoTs output by 
DeepScaleR will be written to SFT_DATASET_FOLDER. The filename will be deepscaler_training_subsample_0_15000.jsonl.

This JSONL file will have 9968 entries - out of these, we create a new JSONL file that contains only the first 4,000 
entries. Store the name of this JSONL file in the environment variable DSR_COT_FILENAME in .env. We will write these
sequential CoTs in a parallel format as follows.

### Rewriting Sequential Rollouts with Claude Sonnet 4.5

In order to rewrite the sequential CoTs to use parallelism, using Claude Sonnet 4.5, run
the following commands from the main directory:
```
set -a; source .env; set +a
cd create_SFT_dataset_files
python claude_parallel_rewrite_updated_instructions.py
```

Please ensure that .env contains the following environment variables:
- SFT_DATASET_FOLDER (the folder where the sequential CoTs are written in the previous step)
- CLAUDE_PROJECT_ID and CLAUDE_LOCATION - these are the project ID and location for your Google Cloud project
- SFT_REWRITE_RESULTS_FILENAME - the name of the file to which the responses rewritten by Claude will be saved
- DSR_COT_FILENAME - the name of the file containing the sequential CoTs which will be rewritten

To reproduce our results, DSR_COT_FILENAME should only contain the first 4,000 results that
are output by generate_deepscaler_model_dataset_responses_UPDATED.py as mentioned above.

## SFT

### SFT Validation Dataset

Running our SFT experiment requires our SFT dataset, which is obtained in the previous
steps. We also use the AIME 2024 dataset for validation, in JSONL format, which can be
obtained as follows:

```
set -a; source .env; set +a
cd create_SFT_dataset_files
python create_AIME_jsonl_validation.py
```

The path where the validation dataset is saved is set in .env through SFT_DATASET_FOLDER.

### SFT Script

To reproduce our experiment, run the following on a SLURM cluster:
```
cd run_SFT
sbatch main_sft_script.sh
```

To run SFT, please set the following variables in .env:
- HUGGINGFACE_HUB_CACHE
- HF_DATASETS_CACHE
- WANDB_CACHE_DIR
- MAIN_OUTPUT_DIR
- SFT_WANDB_PROJECT
- SFT_RUN_NAME

The SFT script will load the training/validation datasets using SFT_DATASET_FOLDER and SFT_REWRITE_RESULTS_FILENAME.
MAIN_OUTPUT_DIR is the main directory where all checkpoints for all runs will be stored. They will be further
organized within this folder by their wandb project name, and then by their wandb run name.

## RL

All code and scripts related to our RL experiments are in run_RL/verl.

### Creating Training/Validation Datasets in .parquet Format

To create the datasets used for RL, in .parquet format, run the following commands

```
cd run_RL/verl/examples/create_parquet_datasets_for_RL/

# Create datasets in .parquet format with correct prompt for DC-CoT
sbatch run_create_RL_datasets_for_dc_cot.sh

# Create datasets in .parquet format with correct prompt for DeepScaleR-1.5B-Preview
sbatch run_create_RL_datasets_for_deepscaler.sh
```

These will create the training/validation datasets in .parquet format for RL. Please set the RL_PARQUET_DATASETS 
environment variable in .env - then, the DeepScaleR training dataset and AIME 2024 validation dataset will be saved 
in ${RL_PARQUET_DATASETS}/aime_2024_standard_prompt_with_special_tokens.

### DC-CoT - Stages 1 through 4

We next run RL, using the [DeepScaleR training dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset), 
and using AIME 2024 as our validation dataset.

The stages of DC-CoT can be run as follows, in the following order:
```
cd run_RL/verl
sbatch examples/dc_cot_stages/dc_cot_stage_1.sh
sbatch examples/dc_cot_stages/dc_cot_stage_2_part_1.sh
sbatch examples/dc_cot_stages/dc_cot_stage_2_part_2.sh
sbatch examples/dc_cot_stages/dc_cot_stage_3.sh
sbatch examples/dc_cot_stages/dc_cot_stage_4.sh
```

Before running dc_cot_stage_1.sh, please set RL_WANDB_PROJECT and RL_STAGE1_NAME. This script will load
from step 1800 of the SFT checkpoint (using previously set .env values to locate the SFT checkpoint), and
will also load the RL datasets from RL_PARQUET_DATASETS/aime_2024_parallel_prompt_with_special_tokens.
All checkpoints and rollouts during the stage 1 RL run will be saved within 
```
${MAIN_OUTPUT_DIR}/${RL_WANDB_PROJECT}/${RL_STAGE1_NAME}
```

The scripts for stages 2 (parts 1 and 2), 3 and 4 follow a similar structure. For these runs, please set
the following variables in .env respectively:
- RL_STAGE2_PART1_NAME
- RL_STAGE2_PART2_NAME
- RL_STAGE3_NAME
- RL_STAGE4_NAME

Each run will continue from a previous run as follows:
- Stage 2 part 1 begins from step 700 of stage 1.
- Stage 2 part 2 begins from step 200 of stage 2 part 1.
- Stage 3 begins from step 280 of stage 2 part 2.
- Stage 4 begins from step 240 of stage 3.

We use FSDP during our RL runs, and checkpoints are saved in a sharded format. To
convert these checkpoints into HF format, use run_RL/verl/scripts/model_merger.py.
Each of our scripts for DC-CoT, in each stage, expects the checkpoint from the previous
stage in HF format.

### HLP Stage - DC-CoT-HLP and DSR-HLP-12K/DSR-HLP-24K

Our scripts for running HLP, starting from DeepScaleR-1.5B-Preview and DC-CoT respectively,
follow a similar structure to the scripts described above, and are located in
run_RL/verl/examples/HLP.

Prior to running dsr_hlp_24k.sh and dsr_hlp_12k.sh, we must explicitly set the padding token
of DeepScaleR-1.5B-Preview. This is necessary in order to ensure that the number of truncated
responses is logged accurately during training. The script create_deepscaler_with_padding_token.py,
located in run_RL/verl/examples/HLP, performs this role, defining the padding token and saving
the updated model/tokenizer to a certain directory - the initial checkpoint is then loaded from
this directory in dsr_hlp_24k.sh and dsr_hlp_12k.sh.

Please run this script as follows:
```
set -a; source .env; set +a
cd run_RL/verl/examples/HLP
python create_deepscaler_with_padding_token.py
```

Afterwards, the HLP scripts can be run as follows, starting from the main directory:
```
cd run_RL/verl/
sbatch examples/HLP/dsr_hlp_12k.sh
sbatch examples/HLP/dsr_hlp_24k.sh
sbatch examples/HLP/dc_cot_hlp.sh
```

## Compute Requirements

For all experiments, we use 1 node with either 8 A100 GPUs or 8 H100 GPUs.

## Citation

If you find our paper or repo useful, please cite our work as follows:

```
@misc{mahankali2026divideandconquercotrlreducing,
      title={Divide-and-Conquer CoT: RL for Reducing Latency via Parallel Reasoning}, 
      author={Arvind Mahankali and Kaiyue Wen and Tengyu Ma},
      year={2026},
      eprint={2601.23027},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.23027}, 
}
```

## License

- This repo uses the MIT License.
- However, the contents of run_RL are based on the [verl package](https://github.com/verl-project/verl), which uses the Apache License 2.0.
- Our base model, DeepScaleR-1.5B-Preview, also uses the MIT License.