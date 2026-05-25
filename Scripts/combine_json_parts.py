import json
import os

def combine_generation_jsons(file_paths, output_path):
    """
    Safely merges multiple generation JSON files into a single master dictionary.
    """
    # Initialize the empty master structure
    combined_data = {
        "human_texts": [],
        "invink_texts": {},
        "weakinvink_texts": {},
        "invink_x_rnm_texts": {}
    }
    
    print("--- STARTING DATA MERGE ---")
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: Could not find {file_path}. Skipping...")
            continue
            
        print(f"Loading data from: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # 1. Append Human Texts
        combined_data["human_texts"].extend(data["human_texts"])
        
        # 2. Append Base InvInk Texts per Epsilon
        for eps, texts in data["invink_texts"].items():
            if eps not in combined_data["invink_texts"]:
                combined_data["invink_texts"][eps] = []
            combined_data["invink_texts"][eps].extend(texts)

        # 3. Append Weak InvInk Texts per Epsilon
        for eps, texts in data["weakinvink_texts"].items():
            if eps not in combined_data["weakinvink_texts"]:
                combined_data["weakinvink_texts"][eps] = []
            combined_data["weakinvink_texts"][eps].extend(texts)
            
        # 4. Append RNM InvInk Texts per Epsilon
        for eps, texts in data["invink_x_rnm_texts"].items():
            if eps not in combined_data["invink_x_rnm_texts"]:
                combined_data["invink_x_rnm_texts"][eps] = []
            combined_data["invink_x_rnm_texts"][eps].extend(texts)
            
    # --- Validation Check ---
    # This prints out the final counts so you can verify no data was lost
    print("\n--- MERGE SUMMARY ---")
    total_human = len(combined_data["human_texts"])
    print(f"Total Human Texts: {total_human}")
    
    for eps in combined_data["invink_texts"]:
        count_base = len(combined_data['invink_texts'][eps])
        count_weak = len(combined_data['weakinvink_texts'][eps])
        count_rnm = len(combined_data['invink_x_rnm_texts'][eps])
        print(f"Epsilon {eps} -> InvInk: {count_base} samples | Weak InvInk: {count_weak} samples | RNM: {count_rnm} samples")
        
        if count_base != total_human or count_weak != total_human or count_rnm != total_human:
            print(f"Mismatch detected at epsilon {eps}! Check your source files.")
            
    # Save the final file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
        
    print(f"\nSuccess! Combined data safely stored at: {output_path}")

# ==========================================
# EXECUTE THE MERGE
# ==========================================

# Put the exact names of the files you uploaded to your Kaggle working directory here
files_to_combine = [
    "Generated_Texts/generated_fine_tuned_texts_experiment_tab_part1.json",
    "Generated_Texts/generated_fine_tuned_texts_experiment_tab_part2.json",
    "Generated_Texts/generated_fine_tuned_texts_experiment_tab_part3.json", 
    "Generated_Texts/generated_fine_tuned_texts_experiment_tab_part4.json",
    "Generated_Texts/generated_fine_tuned_texts_experiment_tab_part5.json"
    # "generated_texts_experiment_part6.json", 
    # "generated_texts_experiment_part7.json",
    # "generated_texts_experiment_part8.json",
    # "generated_texts_experiment_part9.json", 
    # "generated_texts_experiment_part10.json"
]

output_filename = "Generated_Texts/combined_500_fine_tuned_samples_experiment_tab.json"

combine_generation_jsons(files_to_combine, output_filename)