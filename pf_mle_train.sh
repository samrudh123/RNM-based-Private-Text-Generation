#!/bin/bash
#SBATCH --job-name=train_pf_mle
#SBATCH --output=train_mle_output_%j.txt
#SBATCH --partition=gpu-20gb
#SBATCH --gres=gpu:1
#SBATCH --mem=20G

export HF_TOKEN="<your_huggingface_token>"
export HF_HOME="<your_huggingface_cache_directory>"
unset HF_HUB_DISABLE_IMPLICIT_TOKEN

<venv>/bin/python3 -u "Scripts/pf_mle_train.py"