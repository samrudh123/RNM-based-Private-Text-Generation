"""
Load PF-MLE fine-tuned Llama model.
Run this AFTER pf_mle_train.py finishes.

Usage:
    python load_pf_model.py

This script:
  1. Loads base Llama-3.2-3B-Instruct
  2. Applies the LoRA weights from training
  3. Merges them into the base model
  4. Saves a standalone model (no PEFT dependency needed to use it)
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LORA_PATH = "./pf_mle_lora/checkpoint-2000"       # from pf_mle_train.py, add the necessary checkpoint number if you want a different one
MERGED_PATH = "./pf_mle_merged"          # standalone merged model
# HF_TOKEN = os.environ.get("HF_TOKEN", None)

print(f"Loading base model: {MODEL_NAME}")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto"
)

print(f"Loading LoRA weights: {LORA_PATH}")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

print("Merging LoRA into base weights...")
model = model.merge_and_unload()

print(f"Saving merged model to: {MERGED_PATH}")
os.makedirs(MERGED_PATH, exist_ok=True)
model.save_pretrained(MERGED_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(MERGED_PATH)

print(f"""
Done! The merged model is at: {os.path.abspath(MERGED_PATH)}/

You can now use it anywhere without the peft library:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("{MERGED_PATH}")
    tokenizer = AutoTokenizer.from_pretrained("{MERGED_PATH}")

Or in InvisibleInk, just replace the model path:
    MODELS = {{
        'pf_mle': '{os.path.abspath(MERGED_PATH)}',
    }}
""")
