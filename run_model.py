
from typing import Dict, List, Any
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    Trainer
)
from functools import partial
from peft import LoraConfig, PeftModel
import yaml
from tqdm import tqdm
import pandas as pd

from processing_maira2 import Maira2Processor
from train import RadiologyDataset, load_samples_from_jsonl, collate_fn
NUM_SAMPLES = 10
SUBSET_SIZE = None
task = "findings"

def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def preprocess_data(config: Dict, subset_size: int = None):
    data_dir = Path(config["data"]["data_dir"])
    cache_dir = data_dir / config["data"]["cache_dir"]
    test_dataset_path = cache_dir / "test_dataset_findings.jsonl"  # "test_dataset_findings.jsonl"
    print("Use lazy preprocessing...")
    def load_raw(path):
        raw_samples = load_samples_from_jsonl(str(path))
        if subset_size:
            raw_samples = raw_samples[:subset_size]
        return raw_samples

    test_samples = load_raw(test_dataset_path)
    test_dataset = RadiologyDataset(test_samples, config, lazy=True)
    return test_dataset

def run_inference(config: Dict, test_dataset: RadiologyDataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set padding_side to left
    processor = Maira2Processor.from_pretrained(
        config["model"]["name"],
        trust_remote_code=True,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        trust_remote_code=True,
        device_map="auto",
    )
    model.config.use_cache = False
    # load peft model
    peft_config = LoraConfig.from_pretrained(config["inference"]["peft_name"])
    peft_config.base_model_name_or_path = config["model"]["name"]
    model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, config["inference"]["peft_name"]).to(device)
    model.eval()
    torch.cuda.empty_cache()
    
    # process inputs
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["inference"]["batch_size"],
        shuffle=False,
        collate_fn=partial(collate_fn, processor=processor, config=config, dataset="test"),
    )
    list_original, list_struct_ref, list_struct_gen = [], [], []

    # generate structured reports
    # add a progress bar
    progress_bar = tqdm(test_loader, desc="Generating predictions", unit="batch")
    num_batches = 0
    for batch in progress_bar:
        num_batches += 1
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                do_sample=True,
                num_beams=config["inference"]["num_beams"],
                max_new_tokens=config["inference"]["max_new_tokens"],
            )

        list_original.extend(
            processor.batch_decode(batch["input_ids"], skip_special_tokens=True)
        )
        gen = processor.batch_decode(outputs, skip_special_tokens=True)
        try:
            gen = [g.split("ASSISTANT:")[1] for g in gen]
        except:
            print("Could not find ASSISTANT in the generated text")

        list_struct_gen.extend(gen)

        ref_labels = batch["labels"]
        ref_labels[ref_labels == -100] = 0
        list_struct_ref.extend(processor.batch_decode(ref_labels, skip_special_tokens=True))
        
        if (num_batches*config["inference"]["batch_size"]) >= NUM_SAMPLES:
            break

    return list_original,list_struct_gen, list_struct_ref

def main():
    config_path = "config.yaml"
    config = load_config(config_path)

    test_dataset = preprocess_data(
        config,
        subset_size=SUBSET_SIZE)
    list_inp, list_gen, list_refs = run_inference(config, test_dataset)
    print("input:",list_inp[0])
    print("output:",list_gen[0])
    print("reference:",list_refs[0])
    # save as csv
    pd.DataFrame({"input": list_inp, "generated": list_gen, "reference": list_refs}).to_csv("generated_reports.csv", index=False)
    print("Saved generated reports to generated_reports.csv")


if __name__ == "__main__":
    main()