import os
import jsonlines
from datasets import load_dataset

aime_dataset_source = "HuggingFaceH4/aime_2024"
dest_main_dir = os.getenv("SFT_DATASET_FOLDER")
assert dest_main_dir is not None, "Set SFT_DATASET_FOLDER in .env"
dest_filename = "aime_2024_validation.jsonl"

aime_dataset = load_dataset(aime_dataset_source, split="train")

output_path = os.path.join(dest_main_dir, dest_filename)
with jsonlines.open(output_path, "w") as writer:
    for i in range(len(aime_dataset)):
        writer.write(aime_dataset[i])