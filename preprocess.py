# preprocess.py

import json
from PIL import Image
from typing import Dict, List, Any
import os
import logging
from pathlib import Path
import yaml
import random
import torch
from tqdm import tqdm 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUBSET_SIZE = None

def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_samples_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file containing radiology metadata.
    """
    samples = []
    logger.info(f"Loading samples from {jsonl_path}")
    with open(jsonl_path, "r") as file:
        for line in tqdm(file, desc="Reading JSONL file", unit=" lines"):
            sample = json.loads(line)
            image_paths = sample.get("image_paths", [])
            if len(image_paths) > 0:
                sample["frontal_image"] = image_paths[0]
                sample["lateral_image"] = image_paths[1] if len(image_paths) > 1 else None
            else:
                raise ValueError(f"No image paths provided in sample: {sample}")
            samples.append(sample)
    logger.info(f"Loaded {len(samples)} samples.")
    return samples

def load_image(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(image_path, "rb") as f:
        img = Image.open(f)
        img.load()
        return img.convert("RGB")

def preprocess_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess a single sample from the JSONL data.
    """
    frontal_image = load_image(sample["frontal_image"])
    lateral_image = (
        load_image(sample["lateral_image"]) if sample["lateral_image"] else None
    )

    return {
        "frontal_image_path": sample["frontal_image"],
        "lateral_image_path": sample["lateral_image"],
        "indication": sample.get("history_section", ""),
        "technique": sample.get("technique_section", ""),
        "comparison": sample.get("comparison_section", ""),
        "findings": sample.get("findings_section", ""),
    }

def preprocess_dataset(jsonl_path: str, subset_size: int = None) -> List[Dict[str, Any]]:
    """
    Preprocess the dataset from a JSONL file.
    """
    logger.info(f"Loading raw samples from {jsonl_path}")
    raw_samples = load_samples_from_jsonl(jsonl_path)
    logger.info(f"Loaded {len(raw_samples)} raw samples.")

    if subset_size:
        raw_samples = raw_samples[:subset_size]
        logger.info(f"Using subset of size: {subset_size}")

    processed_samples = []
    logger.info("Starting preprocessing of samples...")
    for idx, sample in enumerate(tqdm(raw_samples, desc="Preprocessing samples", unit=" samples")):
        processed_samples.append(preprocess_sample(sample))

    return processed_samples

def save_preprocessed_data(processed_samples: List[Dict[str, Any]], output_path: str):
    """
    Save the preprocessed data to a JSONL file.
    """
    logger.info(f"Saving preprocessed data to {output_path}")
    with open(output_path, "w") as f:
        for sample in tqdm(processed_samples, desc="Writing to JSONL", unit=" samples"):
            json_line = json.dumps(sample)
            f.write(json_line + "\n")
    logger.info("Preprocessing completed and data saved.")

def main():
    config_path = "config.yaml"
    config = load_config(config_path)

    data_dir = Path(config["data"]["data_dir"])
    cache_dir = data_dir / config["data"]["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl_path = cache_dir / "train_dataset.jsonl"
    val_jsonl_path = cache_dir / "val_dataset.jsonl"

    train_output_path = cache_dir / "train_preprocessed.jsonl"
    val_output_path = cache_dir / "val_preprocessed.jsonl"

    # Preprocess training data
    logger.info("Preprocessing training data...")
    train_processed = preprocess_dataset(str(train_jsonl_path), subset_size=SUBSET_SIZE)
    save_preprocessed_data(train_processed, str(train_output_path))

    # Preprocess validation data
    logger.info("Preprocessing validation data...")
    val_processed = preprocess_dataset(str(val_jsonl_path), subset_size=SUBSET_SIZE)
    save_preprocessed_data(val_processed, str(val_output_path))

if __name__ == "__main__":
    main()
