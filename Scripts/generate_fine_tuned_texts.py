import gc
import transformers
transformers.logging.set_verbosity_warning()
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'invisibleink/src'))
sys.path.append(os.path.join(parent_dir, 'invisibleink_x_RNM/src'))
sys.path.append(os.path.join(parent_dir, 'Weak_invisible_ink/src'))

import json
import math
import invink
import invinkxrnm
import weakinvink
from datasets import load_dataset
import torch

part = 5
# 1. Setup and Data Loading
data_tab = load_dataset('mattmdjaga/text-anonymization-benchmark-train')
data_yelp = load_dataset('yelp_review_full', split='train')

human_texts = data_tab['train']['text']
# human_texts = data_yelp.filter(lambda x: x['label'] == part-1)

# model_name = 'meta-llama/Llama-3.2-3B-Instruct'
model_name = '../pf_mle_merged'
dataset_desc_tab = "The dataset comprises English-language court cases from the European Court of Human Rights (ECHR)."
dataset_desc_yelp = "The dataset consists of English-language reviews from Yelp, a popular platform for user-generated content about local businesses."

num_samples = 100
epsilons = [1.0, 5.0, 10.0, 20.0]

# Truncate to the exact number of samples you need
human_texts_sample = human_texts[(part-1)*num_samples:part*num_samples]
# human_texts_sample = list(human_texts.select(range((part-1)*num_samples, part*num_samples))['text'])

reference_pool = human_texts[500:1000]
# reference_pool = list(human_texts.select(range(500, 1000))['text'])

invink_gen_texts_dict = {eps: [] for eps in epsilons}
weakinvink_gen_texts_dict = {eps: [] for eps in epsilons}
invink_x_rnm_gen_texts_dict = {eps: [] for eps in epsilons}

gc.collect() 
torch.cuda.empty_cache()
print("--- STARTING TEXT GENERATION ---")

for eps in epsilons:
    print(f"\n========================================")
    print(f"Processing Epsilon = {eps}")
    print(f"========================================")

    sample_with_references = human_texts_sample + reference_pool

    print("Generating texts with INVINK...")
    # 1. INVINK GENERATION
    output = invink.generate(
        sample_with_references, 
        model_name,
        dtype="float16",
        num=len(human_texts_sample), # Process only the sample
        epsilon=eps,
        topk=100,
        max_toks=256, 
        dataset_desc=dataset_desc_tab, 
        print_text=False,
        batch_size=4
    )
    invink_gen_texts_dict[eps].extend(output.texts)
    
    del output
    gc.collect() 
    torch.cuda.empty_cache()
    
    print("Generating texts with Weak INVINK...")
    # 2. WEAK INVINK GENERATION
    output_weak = weakinvink.generate(
        sample_with_references, 
        model_name,
        dtype="float16",
        num=len(human_texts_sample), # Process only the sample
        epsilon=eps,
        topk=100,
        max_toks=256, 
        dataset_desc=dataset_desc_tab, 
        print_text=False,
        batch_size=4
    )
    weakinvink_gen_texts_dict[eps].extend(output_weak.texts)

    del output_weak
    gc.collect() 
    torch.cuda.empty_cache()

    print("Generating texts with INVINK x RNM...")
    # 3. RNM GENERATION
    output_rnm = invinkxrnm.generate(
        sample_with_references, 
        model_name,
        dtype="float16",
        num=len(human_texts_sample), # Process only the sample
        epsilon=eps,
        topk=100,
        max_toks=256, 
        dataset_desc=dataset_desc_tab, 
        print_text=False,
        batch_size=2
    )
    invink_x_rnm_gen_texts_dict[eps].extend(output_rnm.texts)
    
    del output_rnm
    gc.collect() 
    torch.cuda.empty_cache()

export_data = {
    "human_texts": human_texts_sample,
    "invink_texts": invink_gen_texts_dict,
    "weakinvink_texts": weakinvink_gen_texts_dict,
    "invink_x_rnm_texts": invink_x_rnm_gen_texts_dict
}

save_path = f"generated_fine_tuned_texts_experiment_tab_part{part}.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(export_data, f, ensure_ascii=False, indent=4)

print(f"Success! Data saved to {save_path}")
