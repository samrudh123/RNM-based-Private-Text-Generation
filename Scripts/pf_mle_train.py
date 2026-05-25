"""
=============================================================================
PF-MLE Fine-tuning: Llama-3.2-3B-Instruct with QLoRA
=============================================================================
Run:   python pf_mle_train.py
GPU:   A100 (20+ GB VRAM)
Time:  ~2-4 hours
Output: ./pf_mle_lora/final/  (LoRA adapter weights)

Usage after training:
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    model = PeftModel.from_pretrained(base, "./pf_mle_lora/final/")
    model = model.merge_and_unload()
=============================================================================
"""

import os
import sys
import time
import math
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pf_mle_train.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# =============================================================================
# Config — A100 optimized for 3B
# =============================================================================
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DTYPE = torch.bfloat16

# LoRA
LORA_R = 16               # 3B model is larger → rank 16 is sufficient and saves memory
LORA_ALPHA = 32            # 2 * r
LORA_DROPOUT = 0.0
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
#               ^^^^^ attention only; 3B + FFN would push memory too tight on 20GB

# PF Loss
TAU = 1.1                 # match your InvisibleInk decoding temperature
K_QUAD = 32               # quadrature nodes (verified sufficient)
K_CHUNK = 4               # smaller chunks for 3B — model itself uses more VRAM

# Training
SEQ_LEN = 256
BATCH_SIZE = 1            # PF loss is still heavy at V=128k
GRAD_ACCUM = 8            # effective batch = 8
N_POSITIONS = 2           # PF loss at 2 random positions per sequence (3B needs more VRAM)
LR = 1e-4                 # slightly lower LR for larger model — more stable
WARMUP_STEPS = 100
MAX_STEPS = 5000
SAVE_EVERY = 1000
LOG_EVERY = 25

# Paths
OUTPUT_DIR = "./pf_mle_lora"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# =============================================================================
# PF-MLE Loss Function
# =============================================================================
def log1mexp(x):
    """Stable log(1 - exp(x)) for x <= 0."""
    return torch.where(
        x < -0.6931,
        torch.log1p(-x.exp()),
        torch.log(-torch.expm1(x.clamp(max=-1e-30))),
    )


class PFLoss(nn.Module):
    """
    PF-MLE loss with K-chunking for large vocab (V=128k).
    Returns per-sample loss (unreduced).
    """

    def __init__(self, tau, K=32, k_chunk=8):
        super().__init__()
        u, w = np.polynomial.legendre.leggauss(K)
        u = 0.5 * (u + 1.0)
        w = 0.5 * w
        self.register_buffer("u", torch.tensor(u, dtype=torch.float64))
        self.register_buffer("logu", torch.tensor(np.log(u), dtype=torch.float64))
        self.register_buffer("logw", torch.tensor(np.log(w), dtype=torch.float64))
        self.tau = tau
        self.K = K
        self.k_chunk = k_chunk

    def forward(self, logits, targets):
        logits = logits.double()
        v_max, _ = logits.max(dim=-1, keepdim=True)
        log_p = (logits - v_max) / self.tau
        B, V = logits.shape
        mask = torch.ones_like(logits).scatter_(1, targets.unsqueeze(1), 0.0)

        log_int_pieces = []
        for k0 in range(0, self.K, self.k_chunk):
            k1 = min(k0 + self.k_chunk, self.K)
            logu_chunk = self.logu[k0:k1].view(1, 1, -1)
            log_pu = log_p.unsqueeze(-1) + logu_chunk
            log_term = log1mexp(log_pu)
            inner = (log_term * mask.unsqueeze(-1)).sum(dim=1)
            log_int_pieces.append(self.logw[k0:k1] + inner)

        log_int = torch.logsumexp(torch.cat(log_int_pieces, dim=-1), dim=-1)
        v_y = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((v_y - v_max.squeeze(-1)) / self.tau) - log_int
        return loss.float()


# =============================================================================
# Dataset
# =============================================================================
class TextDS(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        c = self.chunks[idx]
        return {"input_ids": c[:-1], "labels": c[1:]}


def prepare_dataset(tokenizer):
    log.info("Loading WikiText-103 train split...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    all_text = "\n".join([t for t in ds["text"] if len(t.strip()) > 0])
    log.info("Tokenizing (this takes a minute)...")
    all_ids = tokenizer.encode(all_text, add_special_tokens=False)
    log.info(f"Total tokens: {len(all_ids):,}")

    max_chunks = MAX_STEPS * BATCH_SIZE * GRAD_ACCUM * 2
    n_chunks = min(len(all_ids) // (SEQ_LEN + 1), max_chunks)
    chunks = []
    for i in range(n_chunks):
        start = i * (SEQ_LEN + 1)
        chunks.append(torch.tensor(all_ids[start : start + SEQ_LEN + 1], dtype=torch.long))

    log.info(f"Created {len(chunks)} chunks of {SEQ_LEN+1} tokens")
    return TextDS(chunks)


# =============================================================================
# Two-Phase Training Step (memory-safe)
#
# Why two phases?
#   The PF loss creates [B, V, k_chunk] tensors in float64 (V=128k).
#   If we hold the model's backward graph AND the PF loss graph at
#   the same time, we OOM even on A100.
#
# Solution:
#   Phase 1: Forward through model → get logits (graph alive)
#   Phase 2: PF loss on DETACHED logits → get dL/d(logits)
#            (model graph NOT in memory during this)
#   Phase 3: logits.backward(gradient=dL/d_logits)
#            (PF graph NOT in memory during this)
#
# Mathematically identical to single-phase via chain rule:
#   dL/dθ = (dL/dv) · (dv/dθ)
# =============================================================================
def train_step_two_phase(model, pf_loss_fn, batch, device, n_positions):
    """
    Two-phase training step. Computes PF loss at n_positions random positions.
    Each position is processed independently to bound memory.
    Returns: (mean_loss_value, n_tokens)
    """
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    B, T = input_ids.shape

    # Pick random positions
    positions = torch.randint(0, T, (n_positions,))

    # Forward pass — get logits at all positions
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, use_cache=False)
    all_logits = outputs.logits  # [B, T, V]
    del outputs

    total_loss_val = 0.0

    for i, pos in enumerate(positions):
        pos = pos.item()
        is_last = (i == len(positions) - 1)

        # Extract logits at this position (keeps grad connection to model)
        logits_pos = all_logits[:, pos, :].float()  # [1, V]
        target_pos = labels[:, pos]                   # [1]

        # Phase 2: PF loss on detached logits
        logits_det = logits_pos.detach().requires_grad_(True)
        pf_loss = pf_loss_fn(logits_det, target_pos).mean()
        pf_loss.backward()
        grad_pf = logits_det.grad.clone()
        loss_val = pf_loss.item()
        del logits_det, pf_loss

        # Phase 3: backprop through model using PF gradient
        # retain_graph=True for all positions except the last one,
        # because they all share the same model forward graph
        logits_pos.backward(
            gradient=grad_pf / (n_positions * GRAD_ACCUM),
            retain_graph=not is_last,
        )
        del logits_pos, grad_pf

        total_loss_val += loss_val

    del all_logits
    torch.cuda.empty_cache()

    return total_loss_val / n_positions


# =============================================================================
# Validation
# =============================================================================
@torch.no_grad()
def validate(model, pf_loss_fn, tokenizer, device, n_examples=50):
    """Compare PF NLL vs softmax NLL on validation set."""
    model.eval()
    val_ds = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    val_texts = [t for t in val_ds["text"] if len(t.strip()) > 200][:n_examples]

    pf_nlls, sm_nlls = [], []
    for text in val_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < SEQ_LEN + 1:
            continue
        x = torch.tensor([ids[:SEQ_LEN]], device=device)
        y = torch.tensor([ids[SEQ_LEN]], device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).logits[:, -1, :].float()

        pf_nlls.append(pf_loss_fn(logits, y).item())
        sm_nlls.append(-F.log_softmax(logits / TAU, dim=-1).gather(1, y.unsqueeze(1)).item())

    model.train()
    return {
        "pf_nll": np.mean(pf_nlls),
        "sm_nll": np.mean(sm_nlls),
        "gap": np.mean(pf_nlls) - np.mean(sm_nlls),
        "n": len(pf_nlls),
    }


# =============================================================================
# Main
# =============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "GPU required"
    log.info(f"GPU: {torch.cuda.get_device_name()}")
    log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- HF login ---
    if HF_TOKEN:
        from huggingface_hub import login
        login(HF_TOKEN)
        log.info("Logged into HuggingFace")
    else:
        log.warning("No HF_TOKEN set. Run: export HF_TOKEN=hf_xxxxx")

    # --- Tokenizer ---
    log.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info(f"Vocab size: {len(tokenizer)}")

    # --- Model ---
    log.info(f"Loading model: {MODEL_NAME} (4-bit NF4)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=device,
        token=HF_TOKEN,
        dtype=DTYPE,
    )
    model = prepare_model_for_kbit_training(model)

    # --- LoRA ---
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    log.info(f"GPU memory after model: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # --- PF Loss ---
    pf_loss_fn = PFLoss(tau=TAU, K=K_QUAD, k_chunk=K_CHUNK).to(device)

    # --- Sanity check ---
    log.info("Sanity check: PF loss on dummy input...")
    dummy = torch.randn(1, 1000, device=device)
    dummy_t = torch.tensor([42], device=device)
    log.info(f"  PF loss (V=1000) = {pf_loss_fn(dummy, dummy_t).item():.4f} (expect ~5-8)")

    # --- Dataset ---
    dataset = prepare_dataset(tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )

    # --- Optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    log.info(f"Trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS
    )

    # --- Training ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.train()
    model.gradient_checkpointing_enable()

    step = 0
    accum_loss = 0.0
    accum_count = 0
    t0 = time.time()
    loss_history = []

    log.info("=" * 70)
    log.info(f"TRAINING START")
    log.info(f"  max_steps={MAX_STEPS}  eff_batch={BATCH_SIZE*GRAD_ACCUM}  tau={TAU}")
    log.info(f"  n_positions={N_POSITIONS}  seq_len={SEQ_LEN}  lr={LR}")
    log.info(f"  LoRA r={LORA_R}  alpha={LORA_ALPHA}  targets={LORA_TARGETS}")
    log.info("=" * 70)

    for epoch in range(10):  # breaks at MAX_STEPS
        for batch in dataloader:
            loss_val = train_step_two_phase(
                model, pf_loss_fn, batch, device, N_POSITIONS
            )

            accum_loss += loss_val
            accum_count += 1

            if accum_count % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % LOG_EVERY == 0:
                    avg_loss = accum_loss / GRAD_ACCUM
                    loss_history.append(avg_loss)
                    elapsed = time.time() - t0
                    sps = step / elapsed
                    eta = (MAX_STEPS - step) / max(sps, 1e-6) / 60
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    log.info(
                        f"step {step:>5}/{MAX_STEPS} | "
                        f"loss {avg_loss:.4f} | "
                        f"lr {scheduler.get_last_lr()[0]:.2e} | "
                        f"{sps:.2f} it/s | "
                        f"peak {peak:.1f}GB | "
                        f"eta {eta:.0f}min"
                    )
                    accum_loss = 0.0

                if step % SAVE_EVERY == 0:
                    ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{step}")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    log.info(f"Saved checkpoint: {ckpt_dir}")

                    # Run quick validation at each checkpoint
                    val = validate(model, pf_loss_fn, tokenizer, device, n_examples=30)
                    log.info(
                        f"  VAL: pf_nll={val['pf_nll']:.3f}  "
                        f"sm_nll={val['sm_nll']:.3f}  "
                        f"gap={val['gap']:.3f}  (n={val['n']})"
                    )
                    model.train()
                    model.gradient_checkpointing_enable()

                if step >= MAX_STEPS:
                    break

        if step >= MAX_STEPS:
            break

    # --- Save final ---
    final_dir = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Also save the training config for reproducibility
    config = {
        "model": MODEL_NAME, "tau": TAU, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "lora_targets": LORA_TARGETS, "k_quad": K_QUAD, "seq_len": SEQ_LEN,
        "lr": LR, "max_steps": MAX_STEPS, "grad_accum": GRAD_ACCUM,
        "n_positions": N_POSITIONS, "total_time_min": (time.time() - t0) / 60,
    }
    with open(os.path.join(final_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save loss history
    with open(os.path.join(OUTPUT_DIR, "loss_history.json"), "w") as f:
        json.dump(loss_history, f)

    # --- Final validation ---
    log.info("\n" + "=" * 70)
    log.info("FINAL VALIDATION")
    log.info("=" * 70)
    val = validate(model, pf_loss_fn, tokenizer, device, n_examples=50)
    log.info(f"  PF NLL  (fine-tuned): {val['pf_nll']:.3f}")
    log.info(f"  SM NLL  (fine-tuned): {val['sm_nll']:.3f}")
    log.info(f"  Gap (PF - SM):        {val['gap']:.3f}")
    log.info(f"  n_examples:           {val['n']}")

    total_min = (time.time() - t0) / 60
    log.info(f"\nTotal time: {total_min:.1f} min")
    log.info(f"Weights saved to: {os.path.abspath(final_dir)}/")
    log.info(f"Log saved to:     pf_mle_train.log")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        steps = [LOG_EVERY * (i + 1) for i in range(len(loss_history))]
        plt.plot(steps, loss_history, "b-", alpha=0.7)
        plt.xlabel("Step")
        plt.ylabel("PF-MLE Loss")
        plt.title(f"Training Loss (tau={TAU}, LoRA r={LORA_R})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=140)
        log.info(f"Loss curve: {OUTPUT_DIR}/loss_curve.png")
    except Exception as e:
        log.warning(f"Could not plot: {e}")

    log.info("\n" + "=" * 70)
    log.info("DONE. To use these weights:")
    log.info("")
    log.info("  from transformers import AutoModelForCausalLM")
    log.info("  from peft import PeftModel")
    log.info(f'  base = AutoModelForCausalLM.from_pretrained("{MODEL_NAME}")')
    log.info(f'  model = PeftModel.from_pretrained(base, "{os.path.abspath(final_dir)}")')
    log.info("  model = model.merge_and_unload()")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
