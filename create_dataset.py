# create_dataset.py
import os
import json
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_datasets(config: Dict) -> None:
    data_dir = Path(config['data']['data_dataset'])
    splits_file = data_dir / config['data']['splits_file']
    data_file = data_dir / config['data']['data_file']
    mimic_cxr_dir = Path(config['data']['mimic_cxr_dir'])
    cache_dir = Path(config['data']['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_file, 'r') as f:
        splits = json.load(f) 

    with open(data_file, 'r') as f:
        data = json.load(f) 

    train_data = []
    val_data = []
    missing_images = 0
    total_samples = 0

    print("Processing data and creating datasets...")
    for study_id, split in tqdm(splits.items(), desc="Processing Studies"):
        sample = data.get(study_id)
        if not sample:
            continue  

        image_paths = sample.get("image_paths", [])
        if not image_paths:
            missing_images += 1
            continue  

        # Verify image paths
        valid = True
        corrected_paths = []
        for img_rel_path in image_paths:
            img_path = Path(mimic_cxr_dir) / Path(*Path(img_rel_path).parts[1:])
            corrected_paths.append(str(img_path))
            if not img_path.exists():
                print(f"Missing image: {img_path}")
                valid = False
                break
        if not valid:
            missing_images += 1
            continue

        sample_data = {
            "study_id": study_id,  
            "image_paths": corrected_paths,
            "history_section": sample.get("history_section", ""),
            "technique_section": sample.get("technique_section", ""),
            "comparison_section": sample.get("comparison_section", ""),
            "original_report": sample.get("original_report", ""),
            "impression_section": sample.get("impression_section", "") #"findings_section": sample.get("findings_section", "")
        }


        if split == "train":
            train_data.append(sample_data)
        elif split == "validate":
            val_data.append(sample_data)

        total_samples += 1

    print(f"Total samples processed: {total_samples}")
    print(f"Missing or invalid samples: {missing_images}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Save datasets as JSONL
    train_output_path = cache_dir / "train_dataset_impression.jsonl" #train_dataset_findings.jsonl
    val_output_path = cache_dir / "val_dataset_impression.jsonl" #val_dataset_findings.jsonl

    with open(train_output_path, "w") as f:
        for sample in train_data:
            json_line = json.dumps(sample)
            f.write(json_line + "\n")

    with open(val_output_path, "w") as f:
        for sample in val_data:
            json_line = json.dumps(sample)
            f.write(json_line + "\n")

    print(f"Datasets saved as JSONL to {cache_dir}")

if __name__ == "__main__":
    config_path = "config.yaml"  
    config = load_config(config_path)
    create_datasets(config)