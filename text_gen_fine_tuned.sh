#!/bin/bash
#SBATCH --job-name=inv_ink_rnm
#SBATCH --output=fine_tuned_gen_output_%j.txt
#SBATCH --partition=gpu-20gb
#SBATCH --gres=gpu:1
#SBATCH --mem=20G

export HF_TOKEN="<your_huggingface_token>"
export HF_HOME="<your_huggingface_cache_directory>"
unset HF_HUB_DISABLE_IMPLICIT_TOKEN

<venv>/bin/python3 -u Scripts/generate_fine_tuned_texts.py
