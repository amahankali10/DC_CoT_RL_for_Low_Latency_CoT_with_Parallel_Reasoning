###### Function to check the format
worker_start_strings = ["<worker_1>", "<worker_2>", "<worker_3>"]
worker_end_strings = ["</worker_1>", "</worker_2>", "</worker_3>"]
worker_block_start = "<spawn_workers>"
worker_block_end = "</spawn_workers>"
think_start = "<think>"
think_end = "</think>"
answer_start = "<answer>"
answer_end = "</answer>"

def check_parallel_worker_format(completion: str) -> bool:
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