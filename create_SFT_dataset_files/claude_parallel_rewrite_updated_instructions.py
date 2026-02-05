"""
Rewrite in parallel style using new_prompt_with_in_context_example.txt as instructions.
"""

import os
from google.cloud import aiplatform
from anthropic import AnthropicVertex
import jsonlines
import json
from verl_math_verify import verl_math_verify_compute_score
from multiprocessing import Pool
import time

PROJECT_ID = os.getenv("CLAUDE_PROJECT_ID")
LOCATION = os.getenv("CLAUDE_LOCATION")
assert PROJECT_ID is not None, "Set environment variable CLAUDE_PROJECT_ID"
assert LOCATION is not None, "Set environment variable CLAUDE_LOCATION"
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Options
num_rejection_sampling_attempts = 3
num_threads = 50
output_folder = os.getenv("SFT_DATASET_FOLDER")
output_filename = os.getenv("SFT_REWRITE_RESULTS_FILENAME")
assert output_folder is not None, "Set environment variable SFT_DATASET_FOLDER"
assert output_filename is not None, "Set environment variable SFT_REWRITE_RESULTS_FILENAME"
max_tokens = 12000
original_responses_path = os.getenv("DSR_COT_FILENAME")
assert original_responses_path is not None, "Set environment variable DSR_COT_FILENAME"
original_responses_path = os.path.join(output_folder, original_responses_path)

def get_claude_response(prompt: str, system: str = "", temperature: float = 0.0, max_tokens: int = 8192) -> str:
    """
    Get response from Claude using AnthropicVertex client.
    
    Args:
        prompt: The user's input prompt
        system: Optional system message to guide Claude's behavior
        temperature: Controls randomness (0.0 for deterministic, higher for more creative)
    
    Returns:
        The response text from Claude
    """
    client = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)
    
    try:
        message = client.messages.create(
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            system=system,
            temperature=temperature,
            model="claude-sonnet-4-5@20250929"
        )
        
        return message.content[0].text
    except Exception as e:
        print(f"Error: {e}")
        return ""

###### Function to check the format
worker_start_strings = ["<worker_1>", "<worker_2>", "<worker_3>"]
worker_end_strings = ["</worker_1>", "</worker_2>", "</worker_3>"]
worker_block_start = "<spawn_workers>"
worker_block_end = "</spawn_workers>"
think_start = "<think>"
think_end = "</think>"
answer_start = "<answer>"
answer_end = "</answer>"

def check_format(completion: str) -> bool:
    """
    The format of the completion is as follows:
    <think>
    <spawn_workers>
    <worker_1> some text here </worker_1>
    <worker_2> some text here </worker_2>
    <worker_3> some text here </worker_3>
    </spawn_workers>
    some text here.
    <spawn_workers>
    <worker_1> some text here </worker_1>
    <worker_2> some text here </worker_2>
    <worker_3> some text here </worker_3>
    </spawn_workers>
    some text here.
    ... an arbitrary number of times ...
    some text here
    </think>
    <answer>
    some text here
    </answer>

    Notes:
    - The <think> and <answer> tags must be present.
    - The <think></think> and <answer></answer> blocks must be disjoint.
    - Within the <spawn_workers></spawn_workers> blocks, there should not be any text outside of the <worker_1></worker_1>, <worker_2></worker_2>, <worker_3></worker_3> blocks.
    - There should not be any text that is not within the <think></think> or <answer></answer> blocks.
    - Worker blocks must appear in order (1,2,3) and each worker tag must appear exactly once in each <spawn_workers></spawn_workers> block.
    """
    # Check if both think and answer blocks are present
    if think_start not in completion or think_end not in completion:
        return False
    if answer_start not in completion or answer_end not in completion:
        return False
    
    # Check if think and answer blocks are properly ordered and disjoint
    think_start_idx = completion.find(think_start)
    think_end_idx = completion.find(think_end)
    answer_start_idx = completion.find(answer_start)
    answer_end_idx = completion.find(answer_end)
    
    if not (think_start_idx < think_end_idx < answer_start_idx < answer_end_idx):
        return False
    
    # Check that there's no text outside think and answer blocks
    text_before_think = completion[:think_start_idx].strip()
    text_between_blocks = completion[think_end_idx + len(think_end):answer_start_idx].strip()
    text_after_answer = completion[answer_end_idx + len(answer_end):].strip()
    
    if text_before_think or text_between_blocks or text_after_answer:
        return False
    
    # Extract the think block content
    think_content = completion[think_start_idx + len(think_start):think_end_idx].strip()
    
    # Check if there's at least one spawn_workers block in the think content
    if worker_block_start not in think_content or worker_block_end not in think_content:
        return False
    
    # Check each spawn_workers block
    while worker_block_start in think_content:
        # Find the next spawn_workers block
        block_start = think_content.find(worker_block_start)
        block_end = think_content.find(worker_block_end, block_start)
        
        if block_end == -1:  # No matching end tag
            return False
            
        # Extract the block content and remove the opening tag
        block_content = think_content[block_start + len(worker_block_start):block_end].strip()
        
        # Process each worker in order
        for worker_idx in range(3):
            worker_start = worker_start_strings[worker_idx]
            worker_end = worker_end_strings[worker_idx]
            
            # Check if this worker tag exists at the start of the remaining content
            if not block_content.startswith(worker_start):
                return False
                
            # Find the end of this worker block
            worker_end_idx = block_content.find(worker_end)
            if worker_end_idx == -1:
                return False
                
            # Remove this worker block and any whitespace
            block_content = block_content[worker_end_idx + len(worker_end):].strip()
        
        # After processing all workers, block_content should be empty
        if block_content:
            return False
        
        # Move to next spawn_workers block
        think_content = think_content[block_end + len(worker_block_end):].strip()
    
    return True

original_dataset_question_key = "problem"
with open("new_prompt_with_in_context_example.txt", "r") as f:
    claude_prompt_template = f.read()

# Define a worker that processes one contiguous chunk of examples and logs per-thread progress
def process_chunk(thread_id, examples):
    results = []
    C = len(examples)
    for idx, original_dict in enumerate(examples):
        for attempt in range(num_rejection_sampling_attempts):

            claude_response = ""
            backoff = 1
            while len(claude_response) == 0: # Don't want resource exhausted/exception to count towards attempts
                claude_response = get_claude_response(
                    prompt=claude_prompt_template.format(
                        question=original_dict[original_dataset_question_key],
                        assistant_response=original_dict["long_cot_completion"]
                    ),
                    temperature=0.7,
                    max_tokens=max_tokens
                )

                if len(claude_response) == 0:
                    time.sleep(backoff)
                    backoff = 2 * backoff

            if not check_format(claude_response):
                continue
            score = verl_math_verify_compute_score(model_output=claude_response, ground_truth=original_dict["ground_truth_answer"])
            if abs(score - 1.0) < 1e-6:
                print(f"[Thread {thread_id}] Example {idx+1}/{C} ACCEPTED - num attempts: {attempt+1}")
                res = original_dict.copy()
                res["parallel_rewritten_completion"] = claude_response
                results.append(res)
                break
        else:
            print(f"[Thread {thread_id}] Example {idx+1}/{C} SKIPPED - num attempts: {num_rejection_sampling_attempts}")
    return results

# Main program: split data into K roughly equal chunks and process in parallel
if __name__ == "__main__":
    # Load the original responses
    with open(original_responses_path, "r") as f:
        original_responses = [json.loads(line) for line in f]

    total = len(original_responses)
    K = num_threads
    q, r = divmod(total, K)
    chunks = []
    start = 0
    for i in range(K):
        size = q + 1 if i < r else q
        chunks.append(original_responses[start:start+size])
        start += size

    # Dispatch each chunk to a worker process
    args = [(i, chunks[i]) for i in range(K)]
    with Pool(processes=K) as pool:
        thread_results = pool.starmap(process_chunk, args)

    # Flatten all accepted results and write to JSONLines
    processed = [item for sub in thread_results for item in sub]

    # Prepare output directory and file
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, output_filename)
    print(f"Output will be saved to: {output_file}")

    # Write the processed examples to JSONLines
    with jsonlines.open(output_file, mode="w") as writer:
        for result in processed:
            writer.write(result)