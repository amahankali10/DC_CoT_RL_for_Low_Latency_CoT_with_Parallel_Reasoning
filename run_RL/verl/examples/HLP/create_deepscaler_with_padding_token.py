from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Download model and tokenizer
model_name = "agentica-org/DeepScaleR-1.5B-Preview"
tokenizer = AutoTokenizer.from_pretrained(model_name)
lm = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad token
tokenizer.pad_token = "<|fim_pad|>"
tokenizer.pad_token_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]

# Save model and tokenizer
main_dir = os.getenv("MAIN_OUTPUT_DIR")
assert main_dir is not None, "Set MAIN_OUTPUT_DIR in .env"
subfolder = "deepscaler_1_5_b_preview_with_pad_token_in_tokenizer"
output_dir = os.path.join(main_dir, subfolder)
lm.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)