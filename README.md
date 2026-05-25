# Project Setup and Sample Run

## 1) Clone this repository (Project folder)

```bash
git clone https://github.com/samrudh123/RNM-based-Private-Text-Generation.git
cd "RNM-based-Private-Text-Generation"
```

## 2) Clone required repositories inside this repo

```bash
git clone https://github.com/cerai-iitm/invisibleink.git
git clone https://github.com/samrudh123/invisibleink_x_RNM.git
git clone https://github.com/samrudh123/Weak_invisible_ink.git
```

> Replace the placeholder URLs with the actual repository links.

## 3) Set up a virtual environment and install dependencies

Windows (PowerShell):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4) Run sh files to generate the output in the respective folders

After you update each script with your paths/tokens, run them in this order:

1. `pf_mle_train.sh` (train the PF MLE model)
2. `load_lora_model.sh` (load the trained model)
3. `text_gen.sh` (generate base texts)
4. `text_gen_fine_tuned.sh` (generate fine-tuned texts)
5. `combine_json_files.sh` (merge part files into combined JSONs)
6. `mauve_scores.sh` (compute Mauve scores)
7. `perplexity.sh` (compute perplexity differences)

Notes:
- Update `HF_TOKEN`, `HF_HOME`, and the `<venv>` path in each script before running.
- Use `bash <script>.sh` locally or `sbatch <script>.sh` on a SLURM cluster.


