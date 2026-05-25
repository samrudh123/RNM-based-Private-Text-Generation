import json
import mauve
import matplotlib.pyplot as plt
import torch

# data_path = "generated_texts_experiment_part1.json" 
data_path = "Generated_Texts/combined_500_fine_tuned_samples_experiment_tab.json"

print(f"Loading data from: {data_path}")
with open(data_path, "r", encoding="utf-8") as f:
    imported_data = json.load(f)

# Extract human texts directly from the JSON
human_texts_sample = imported_data["human_texts"]

# JSON saves dictionary keys as strings, so we map them back to floats
invink_texts_dict = {
    float(k): v for k, v in imported_data["invink_texts"].items()
    }
weakinvink_texts_dict = {
    float(k): v for k, v in imported_data["weakinvink_texts"].items()
    }
invink_x_rnm_texts_dict = {
    float(k): v for k, v in imported_data["invink_x_rnm_texts"].items()
    }
# invink_x_rnm_pf_texts_dict = {
#     float(k): v for k, v in imported_data["invink_x_rnm_pf_texts"].items()
#     }

epsilons = sorted(list(invink_texts_dict.keys()))
print("\n--- STARTING MAUVE EVALUATION ---")
invink_mauve_scores = []
weakinvink_mauve_scores = []
rnm_mauve_scores = []
# rnm_pf_mauve_scores = []

device_id = 0 if torch.cuda.is_available() else -1

for eps in epsilons:
    print(f"\nComputing MAUVE for epsilon = {eps}...")
    
    print("  -> Evaluating standard InvInk...")
    out_invink = mauve.compute_mauve(
        p_text=human_texts_sample, 
        q_text=invink_texts_dict[eps], 
        device_id=device_id, 
        max_text_length=256, 
        mauve_scaling_factor=2,
        num_buckets=11,
        featurize_model_name='gpt2-xl'
        # Use the same model for featurization to ensure consistency 
    )
    invink_mauve_scores.append(out_invink.mauve)
    print(f"  -> InvInk Score: {out_invink.mauve:.4f}")

    print("  -> Evaluating weak InvInk...")
    out_weakinvink = mauve.compute_mauve(
        p_text=human_texts_sample, 
        q_text=weakinvink_texts_dict[eps], 
        device_id=device_id, 
        max_text_length=256, 
        mauve_scaling_factor=2,
        num_buckets=11,
        featurize_model_name='gpt2-xl'
        # Use the same model for featurization to ensure consistency 
    )
    weakinvink_mauve_scores.append(out_weakinvink.mauve)
    print(f"  -> Weak InvInk Score: {out_weakinvink.mauve:.4f}")

    print("  -> Evaluating InvInk + RNM...")
    out_rnm = mauve.compute_mauve(
        p_text=human_texts_sample, 
        q_text=invink_x_rnm_texts_dict[eps], 
        device_id=device_id, 
        max_text_length=256, 
        mauve_scaling_factor=2,
        num_buckets=11,
        featurize_model_name='gpt2-xl'
        # Use the same model for featurization to ensure consistency 
    )
    rnm_mauve_scores.append(out_rnm.mauve)
    print(f"  -> RNM Score: {out_rnm.mauve:.4f}")

    # print("  -> Evaluating InvInk + RNM + PF...")
    # out_rnm_pf = mauve.compute_mauve(
    #     p_text=human_texts_sample, 
    #     q_text=invink_x_rnm_pf_texts_dict[eps], 
    #     device_id=device_id, 
    #     max_text_length=256, 
    #     mauve_scaling_factor=2,
    #     num_buckets=11,
    #     featurize_model_name='gpt2-xl'
    #     # Use the same model for featurization to ensure consistency 
    # )
    # rnm_pf_mauve_scores.append(out_rnm_pf.mauve)
    # print(f"  -> RNM + PF Score: {out_rnm_pf.mauve:.4f}")

# Plotting the results
print("\n--- PLOTTING RESULTS ---")
plt.figure(figsize=(9, 6))

# Plot both lines for comparison
plt.plot(
    epsilons, invink_mauve_scores, marker='o',
    linestyle='-', color='blue', linewidth=2,
    label='Base Invisible Ink'
    )
plt.plot(
    epsilons, weakinvink_mauve_scores, marker='^',
    linestyle='-.', color='green', linewidth=2,
    label='Weak Invisible Ink'
    )
plt.plot(
    epsilons, rnm_mauve_scores, marker='s',
    linestyle='--', color='red', linewidth=2,
    label='Invisible Ink + RNM'
    )
# plt.plot(
#     epsilons, rnm_pf_mauve_scores, marker='d',
#     linestyle=':', color='purple', linewidth=2,
#     label='Invisible Ink + RNM + PF'
#     )

plt.title(r'MAUVE Score vs Privacy Budget ($\epsilon$) for TAB Dataset')
plt.xlabel(r'Privacy Budget ($\epsilon$)')
plt.ylabel('MAUVE Score')
# plt.ylim(0, 1.0) # MAUVE scores range from 0 to 1
plt.xticks(epsilons) # Align x-ticks cleanly with the evaluated budgets
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')

plt.savefig('Plots/mauve_vs_epsilon_500_fine_tuned_samples_tab.png', bbox_inches='tight')
print("Plot saved successfully as 'mauve_vs_epsilon_500_fine_tuned_samples_tab.png'")