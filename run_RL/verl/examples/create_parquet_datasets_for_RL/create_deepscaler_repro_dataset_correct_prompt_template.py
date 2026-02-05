import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
from enum import Enum
import argparse

import pandas as pd

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string

class ValDatasetPath(Enum):
    AIME_2024 = "HuggingFaceH4/aime_2024"
    MATH_500 = "HuggingFaceH4/MATH-500"
    AMC_23 = "math-ai/amc23"
    MINERVA_MATH = "math-ai/minervamath"
    OLYMPIAD_BENCH = "Hothan/OlympiadBench"
    HMMT_FEB_2025 = "MathArena/hmmt_feb_2025"

prefix = "<｜begin▁of▁sentence｜><｜User｜>"
suffix = "<｜Assistant｜>"
instruction = "Let's think step by step and output the final answer within \\boxed{{}}."
think_token = "<think>\n"
deepseek_r1_prompt = prefix + "{question} " + instruction + suffix + think_token

parallel_instruction = "You can spawn multiple workers to solve this problem in parallel." \
                       " The workers' thoughts are enclosed within <spawn_workers></spawn_workers> tags, and each worker's" \
                       " thought is enclosed within <worker_i></worker_i> tags, where i is the worker number, i.e." \
                       " <spawn_workers><worker_1>worker 1's thought</worker_1><worker_2>worker 2's thought</worker_2>..." \
                       "</spawn_workers>."
deepseek_r1_prompt_parallel = prefix + "{question} " + parallel_instruction + " " + instruction + suffix + think_token

"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

def make_map_fn(split: str, prompt_template=None):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """

    def process_fn(example, idx):
        question = example.pop("problem")

        assert prompt_template is not None
        # if instruction is None:
        #     instruction = "Let's think step by step and output the final answer within \\boxed{}."

        answer = example.pop("answer")

        data = {
            "data_source": "DigitalLearningGmbH/MATH-lighteval", # This is so that the math verify reward function is used
            "prompt": [{"role": "user", "content": prompt_template.format(question=question)}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "task": {"question": question, "ground_truth": answer},
            },
            "task": {"question": question, "ground_truth": answer},
            "uid": idx,
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--use_parallel_prompt', action='store_true')
    parser.add_argument('--val_dataset_path', default=ValDatasetPath.AIME_2024.value)
    args = parser.parse_args()

    train_dataset = datasets.load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
    if args.val_dataset_path == ValDatasetPath.AIME_2024.value:
        test_dataset = datasets.load_dataset(ValDatasetPath.AIME_2024.value, split="train")
    elif args.val_dataset_path == ValDatasetPath.MATH_500.value:
        test_dataset = datasets.load_dataset(ValDatasetPath.MATH_500.value, split="test")
    elif args.val_dataset_path == ValDatasetPath.AMC_23.value:
        test_dataset = datasets.load_dataset(ValDatasetPath.AMC_23.value, split="test")
        test_dataset = test_dataset.rename_column("question", "problem")
    elif args.val_dataset_path == ValDatasetPath.MINERVA_MATH.value:
        test_dataset = datasets.load_dataset(ValDatasetPath.MINERVA_MATH.value, split="test")
        test_dataset = test_dataset.rename_column("question", "problem")
    elif args.val_dataset_path == ValDatasetPath.OLYMPIAD_BENCH.value:
        OLYMPIAD_BENCH_CONFIG = "OE_TO_maths_en_COMP"
        test_dataset = datasets.load_dataset(ValDatasetPath.OLYMPIAD_BENCH.value, OLYMPIAD_BENCH_CONFIG, split="train")
        def clean_olympiad_bench(example):
            example["final_answer"] = example["final_answer"][0]
            return example
        test_dataset = test_dataset.map(clean_olympiad_bench)
        test_dataset = test_dataset.rename_column("question", "problem")
        test_dataset = test_dataset.rename_column("final_answer", "answer")
    elif args.val_dataset_path == ValDatasetPath.HMMT_FEB_2025.value:
        test_dataset = datasets.load_dataset(ValDatasetPath.HMMT_FEB_2025.value, split="train")
    else:
        raise ValueError(f"Invalid validation dataset path: {args.val_dataset_path}")

    if args.use_parallel_prompt:
        prompt_template = deepseek_r1_prompt_parallel
    else:
        prompt_template = deepseek_r1_prompt

    train_dataset = train_dataset.map(function=make_map_fn('train', prompt_template=prompt_template), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test', prompt_template=prompt_template), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train_deepscaler.parquet'))
    if args.val_dataset_path == ValDatasetPath.AIME_2024.value:
        test_dataset.to_parquet(os.path.join(local_dir, 'test_aime.parquet'))
    elif args.val_dataset_path == ValDatasetPath.MATH_500.value:
        test_dataset.to_parquet(os.path.join(local_dir, 'test_math500.parquet'))
    elif args.val_dataset_path == ValDatasetPath.AMC_23.value:
        test_dataset.to_parquet(os.path.join(local_dir, 'test_amc23.parquet'))
    elif args.val_dataset_path == ValDatasetPath.MINERVA_MATH.value:
        test_dataset.to_parquet(os.path.join(local_dir, 'test_minervamath.parquet'))
    elif args.val_dataset_path == ValDatasetPath.OLYMPIAD_BENCH.value:
        test_dataset.to_parquet(os.path.join(local_dir, 'test_olympiadbench.parquet'))
    elif args.val_dataset_path == ValDatasetPath.HMMT_FEB_2025.value:
        test_dataset.to_parquet(os.path.join(local_dir, 'test_hmmt_feb_2025.parquet'))
    else:
        raise ValueError(f"Invalid validation dataset path: {args.val_dataset_path}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)