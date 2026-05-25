import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

model_name = "meta-llama/Llama-3.2-3B-Instruct" # Evaluator model
data_path = "Generated_Texts/combined_500_samples_experiment_tab.json"

print("Loading dataset...")
with open(data_path, "r", encoding="utf-8") as f:
    imported_data = json.load(f)

human_texts = imported_data["human_texts"]
invink_texts_dict = {float(k): v for k, v in imported_data["invink_texts"].items()}
weakinvink_texts_dict = {float(k): v for k, v in imported_data["weakinvink_texts"].items()}
invink_x_rnm_texts_dict = {float(k): v for k, v in imported_data["invink_x_rnm_texts"].items()}
epsilons = sorted(list(invink_texts_dict.keys()))

print(f"Loading Evaluator Model: {model_name}...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# If pad token is missing, assign it to eos_token to prevent errors
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load in float16 to save VRAM on the T4 GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    dtype=torch.float16, 
    device_map="auto"
)
model.eval() # Set to evaluation mode!

def calculate_average_perplexity(texts, model, tokenizer, device):
    """Calculates the average perplexity for a list of texts safely."""
    ppl_scores = []
    
    # Process one text at a time to guarantee zero OOM errors
    for text in texts:
        # Ignore empty strings
        if not text.strip():
            continue
            
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        input_ids = encodings.input_ids.to(device)
        
        # We don't need gradients for evaluation
        with torch.no_grad():
            # Passing labels=input_ids tells Hugging Face to calculate the loss for us!
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            ppl = torch.exp(loss).item()
            
            # Filter out infinite/NaN perplexities (happens if text is gibberish)
            if not np.isnan(ppl) and not np.isinf(ppl):
                ppl_scores.append(ppl)
                
    return np.mean(ppl_scores)

print("\n--- CALCULATING PERPLEXITY ---")

print("Calculating Human Baseline Perplexity...")
human_ppl = calculate_average_perplexity(human_texts, model, tokenizer, device)
print(f"Human PPL: {human_ppl:.2f}\n")

# Store differences: |Human_PPL - Generated_PPL|
invink_ppl_diffs = []
weakinvink_ppl_diffs = []
rnm_ppl_diffs = []
invink_ppls = []
weakinvink_ppls = []
rnm_ppls = []

for eps in epsilons:
    print(f"Processing Epsilon = {eps}...")
    
    # Base InvInk
    invink_ppl = calculate_average_perplexity(invink_texts_dict[eps], model, tokenizer, device)
    invink_ppls.append(invink_ppl)
    diff_invink = abs(human_ppl - invink_ppl)
    invink_ppl_diffs.append(diff_invink)
    print(f"  -> Base InvInk PPL: {invink_ppl:.2f} (Diff: {diff_invink:.2f})")
    
    # Weak InvInk
    weakinvink_ppl = calculate_average_perplexity(weakinvink_texts_dict[eps], model, tokenizer, device)
    weakinvink_ppls.append(weakinvink_ppl)
    diff_weakinvink = abs(human_ppl - weakinvink_ppl)
    weakinvink_ppl_diffs.append(diff_weakinvink)
    print(f"  -> Weak InvInk PPL: {weakinvink_ppl:.2f} (Diff: {diff_weakinvink:.2f})")

    # RNM InvInk
    rnm_ppl = calculate_average_perplexity(invink_x_rnm_texts_dict[eps], model, tokenizer, device)
    rnm_ppls.append(rnm_ppl)
    diff_rnm = abs(human_ppl - rnm_ppl)
    rnm_ppl_diffs.append(diff_rnm)
    print(f"  -> RNM InvInk PPL : {rnm_ppl:.2f} (Diff: {diff_rnm:.2f})")

print("\n--- PLOTTING RESULTS ---")
print("Plotting Perplexity Differences...")
plt.figure(figsize=(9, 6))

plt.plot(epsilons, invink_ppl_diffs, marker='o', linestyle='-', color='blue', linewidth=2, label='Base Invisible Ink')
plt.plot(epsilons, weakinvink_ppl_diffs, marker='^', linestyle='-.', color='green', linewidth=2, label='Weak Invisible Ink')
plt.plot(epsilons, rnm_ppl_diffs, marker='s', linestyle='--', color='red', linewidth=2, label='Invisible Ink + RNM')

plt.title(r'Absolute Perplexity Difference (Human vs. Generated) vs Privacy Budget ($\epsilon$) for TAB dataset')
plt.xlabel(r'Privacy Budget ($\epsilon$)')
plt.ylabel(r'| $\text{PPL}_{\text{Human}}$ - $\text{PPL}_{\text{Gen}}$ |')
plt.xticks(epsilons)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')

# A lower difference is better (meaning generated text is closer to human text)
plt.text(0.02, 0.02, 'Lower is Better (Closer to Human)', transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom')

plt.savefig('perplexity_difference_500_samples_tab.png', bbox_inches='tight')
print("Plot saved successfully as 'perplexity_difference_500_samples_tab.png'")

print("Plotting Raw Perplexities...")
plt.figure(figsize=(10, 6))

# Plot the generated text perplexities
plt.plot(epsilons, invink_ppls, marker='o', linestyle='-', color='blue', linewidth=2, label='Base Invisible Ink')
plt.plot(epsilons, weakinvink_ppls, marker='^', linestyle='-.', color='green', linewidth=2, label='Weak Invisible Ink')
plt.plot(epsilons, rnm_ppls, marker='s', linestyle='--', color='red', linewidth=2, label='Invisible Ink + RNM')

# Plot Human Perplexity as a horizontal target line across the graph
plt.axhline(y=human_ppl, color='green', linestyle=':', linewidth=2.5, label='Human Baseline')

plt.title(r'Raw Perplexity vs Privacy Budget ($\epsilon$) for TAB dataset')
plt.xlabel(r'Privacy Budget ($\epsilon$)')
plt.ylabel('Average Perplexity')
plt.xticks(epsilons)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')

plt.savefig('Plots/raw_perplexity_500_samples_tab.png', bbox_inches='tight')
print("Plot saved successfully as 'raw_perplexity_500_samples_tab.png'")