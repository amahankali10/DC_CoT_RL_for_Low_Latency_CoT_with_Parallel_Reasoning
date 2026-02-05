"""
This script does the following.
- Download the training examples from agentica-org/DeepScaleR-Preview-Dataset
- For each example, generate completions using agentica-org/DeepScaleR-1.5B-Preview
- Generate up to 3 completions, and only include completions that get the correct answer.
- Save the dataset to a JSON file in the desired output folder, with the following fields:
    - problem
    - long_cot_completion
    - ground_truth_solution

The parts related to checking correctness of the math solution are from
verl/verl/utils/reward_score/math.py
"""
import os
import datasets
import jsonlines
from vllm import LLM, SamplingParams
import argparse
from verl_math_verify import verl_math_verify_compute_score

math_dataset_path = "agentica-org/DeepScaleR-Preview-Dataset"
original_dataset_question_key = "problem"
dsr_model_path = "agentica-org/DeepScaleR-1.5B-Preview" # This is the model path
num_attempts_for_correct_answer = 3
output_folder = os.getenv("SFT_DATASET_FOLDER")
if output_folder is None:
    assert False, "Must provide output folder (SFT_DATASET_FOLDER) for DeepScaleR rollouts"
temperature = 0.6
top_p = 0.95
batch_size = -1
max_model_len = 40000
max_tokens = 24000
sampling_seed = 42

instruction = "Let's think step by step and output the final answer within \\boxed{{}}."
deepseek_r1_prompt = "{question} " + instruction + "<think>\n"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample-start", type=int, default=None)
    parser.add_argument("--subsample-end", type=int, default=None)
    args = parser.parse_args()

    dataset = datasets.load_dataset(math_dataset_path)
    dataset = dataset["train"]
    os.makedirs(output_folder, exist_ok=True)
    thoughts_model = LLM(
        model=dsr_model_path,
        tensor_parallel_size=1,  # Adjust based on available GPUs
        gpu_memory_utilization=0.7,
        max_model_len=max_model_len,
        max_num_batched_tokens=40000,
        max_num_seqs=2000,
    )

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=sampling_seed,
    )

    subsample_start = args.subsample_start
    subsample_end = args.subsample_end
    assert subsample_start is not None and subsample_end is not None
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(subsample_start, subsample_end))
    output_filename = f"deepscaler_training_subsample_{subsample_start}_{subsample_end}.jsonl"
    output_file = os.path.join(output_folder, output_filename)
    print(f"Output will be saved to: {output_file}")

    # Process examples in batches
    num_examples = len(dataset)
    if batch_size < 0:
        batch_size = num_examples
    with jsonlines.open(output_file, mode="w") as writer:
        for batch_start in range(0, num_examples, batch_size):
            batch_end = min(batch_start + batch_size, num_examples)
            batch_size_actual = batch_end - batch_start

            # Prepare batch of questions
            batch_questions = []
            batch_original_dict = []
            for i in range(batch_start, batch_end):
                question = dataset[i][original_dataset_question_key]
                text = deepseek_r1_prompt.format(question=question)
                batch_questions.append(text)
                batch_original_dict.append(dataset[i])

            print(f"Processing batch of {batch_size_actual} questions (examples {batch_start+1}-{batch_end})")

            for attempt_idx in range(num_attempts_for_correct_answer):
                print(f"Attempt {attempt_idx+1} of {num_attempts_for_correct_answer} for batch {batch_start+1}-{batch_end} - Num questions: {len(batch_questions)}")

                # Generate completions for all remaining examples in this batch
                outputs = thoughts_model.generate(batch_questions, sampling_params)

                remaining_batch_questions = []
                remaining_batch_original_dict = []

                # Check correctness of each output
                for i, output in enumerate(outputs):
                    original_dict = batch_original_dict[i]
                    long_cot_completion = output.outputs[0].text

                    # Check if long_cot_completion has correct answer
                    ground_truth_answer = original_dict["answer"]
                    is_correct = verl_math_verify_compute_score(long_cot_completion, ground_truth_answer)

                    if is_correct:
                        new_dict = {}
                        new_dict.update(original_dict)
                        new_dict["long_cot_completion"] = long_cot_completion
                        new_dict["ground_truth_answer"] = ground_truth_answer
                        writer.write(new_dict)
                    else:
                        remaining_batch_questions.append(batch_questions[i])
                        remaining_batch_original_dict.append(original_dict)
                
                batch_questions = remaining_batch_questions
                batch_original_dict = remaining_batch_original_dict

                del outputs