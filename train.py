import json
from PIL import Image
from typing import Dict, List, Any
import os
import logging
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
)
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, get_peft_model
import yaml
import random
import wandb
from processing_maira2 import Maira2Processor
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
        for line in file:
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
    # logger.info(f"Preprocessing sample with ID: {sample.get('study_id', 'Unknown ID')}")
    frontal_image = load_image(sample["frontal_image"])
    lateral_image = (
        load_image(sample["lateral_image"]) if sample["lateral_image"] else None
    )

    return {
        "frontal_image": frontal_image,
        "lateral_image": lateral_image,
        "indication": sample.get("history_section", ""),
        "technique": sample.get("technique_section", ""),
        "comparison": sample.get("comparison_section", ""),
        "findings": sample.get("findings_section", ""),  # "findings": sample.get("findings_section", ""),
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
        logger.info(f"Subset size set to {subset_size}. Using first {subset_size} samples.")

    processed_samples = []
    for idx, sample in enumerate(tqdm(raw_samples, desc="Preprocessing samples", unit="sample")):
        # logger.info(f"Processing sample {idx + 1}/{len(raw_samples)}") 
        processed_samples.append(preprocess_sample(sample))

    return processed_samples


class RadiologyDataset(Dataset):
    """Custom Dataset for Radiology Data."""

    def __init__(self, samples: List[Dict[str, Any]], config: Dict, lazy: bool = False):
        self.config = config
        self.lazy = lazy
        if self.lazy:
            self.raw_samples = samples
            logger.info("Initialized Dataset with lazy preprocessing.")
        else:
            self.samples = samples 
            logger.info("Initialized Dataset with preprocessed data.")

    def __len__(self):
        if self.lazy:
            return len(self.raw_samples)
        else:
            return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.lazy:
            sample = self.raw_samples[idx]
            return preprocess_sample(sample)
        else:
            return self.samples[idx]


def collate_fn(batch: List[Dict[str, Any]], processor, config, dataset) -> Dict[str, torch.Tensor]:
    batch_input_ids = []
    batch_attention_mask = []
    batch_pixel_values = []
    batch_labels = []

    for sample in batch:
        # Process inputs without assistant text (user input)
        processed_inputs_no_assistant = processor.format_and_preprocess_reporting_input(
            current_frontal=sample["frontal_image"],
            current_lateral=None,
            prior_frontal=None,
            indication=sample["indication"],
            technique=sample["technique"],
            comparison=sample["comparison"],
            prior_report=None,
            get_grounding=False,
            return_tensors="pt",
        )

        # Process inputs with assistant text (including target)
        processed_inputs_with_assistant = processor.format_and_preprocess_reporting_input(
            current_frontal=sample["frontal_image"],
            current_lateral=None,
            prior_frontal=None,
            indication=sample["indication"],
            technique=sample["technique"],
            comparison=sample["comparison"],
            prior_report=None,
            get_grounding=False,
            assistant_text=sample["findings"],  # sample["findings"],
            return_tensors="pt",
        )

        input_ids_no_assistant = processed_inputs_no_assistant["input_ids"].squeeze(0)
        input_ids_with_assistant = processed_inputs_with_assistant["input_ids"].squeeze(0)
        attention_mask = processed_inputs_with_assistant["attention_mask"].squeeze(0)
        pixel_values = processed_inputs_with_assistant["pixel_values"].squeeze(0)

        # Create labels with user input masked (-100)
        labels = input_ids_with_assistant.clone()
        user_input_length = input_ids_no_assistant.size(0)
        labels[:user_input_length] = -100

        if dataset == "train":
            batch_input_ids.append(input_ids_with_assistant)
        else:
            batch_input_ids.append(input_ids_no_assistant)
        batch_attention_mask.append(attention_mask)
        batch_pixel_values.append(pixel_values)
        batch_labels.append(labels)

    # Pad sequences to the longest in the batch
    batch_input_ids = pad_sequence(
        batch_input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id
    )
    batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
    batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100)
    batch_pixel_values = torch.stack(batch_pixel_values)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
        "pixel_values": batch_pixel_values,
    }


def preprocess_data(config: Dict, subset_size: int = None, lazy_preprocess: bool = False):
    data_dir = Path(config["data"]["data_dir"])
    cache_dir = data_dir / config["data"]["cache_dir"]
    train_dataset_path = cache_dir / "train_dataset_findings.jsonl"  # "train_dataset_findings.jsonl"
    val_dataset_path = cache_dir / "val_dataset_findingsjsonl"  # "val_dataset_findings.jsonl"

    if lazy_preprocess:
        logger.info("Using lazy preprocessing mode.")
        def load_raw(path):
            raw_samples = load_samples_from_jsonl(str(path))
            if subset_size:
                raw_samples = raw_samples[:subset_size]
            return raw_samples

        train_samples = load_raw(train_dataset_path)
        val_samples = load_raw(val_dataset_path)
    else:
        logger.info("Using eager preprocessing mode with a loading bar.")
        def load_and_preprocess(path):
            raw_samples = load_samples_from_jsonl(str(path))
            if subset_size:
                raw_samples = raw_samples[:subset_size]
            processed_samples = preprocess_dataset(str(path), subset_size=subset_size)
            return processed_samples

        train_samples = load_and_preprocess(train_dataset_path)
        val_samples = load_and_preprocess(val_dataset_path)

    train_dataset = RadiologyDataset(train_samples, config, lazy=lazy_preprocess)
    val_dataset = RadiologyDataset(val_samples, config, lazy=lazy_preprocess)
    return train_dataset, val_dataset


class GradientNormCallback(TrainerCallback):
    """
    A custom callback to compute and log gradient norms after each training step.
    """
    def on_step_end(self, args, state, control, **kwargs):
        trainer = kwargs.get('trainer')
        if trainer is None:
            return

        gradient_norms = []

        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                gradient_norms.append((name, param_norm))

        total_norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=float('inf')).item()

        logger.info(f"Total gradient norm: {total_norm}")


class LearningRateLoggerCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        trainer = kwargs.get('trainer')
        if trainer is None:
            return
        current_lr = trainer.lr_scheduler.get_last_lr()[0]
        logger.info(f"Current Learning Rate: {current_lr}")


def train_model(config: Dict, train_dataset: RadiologyDataset, val_dataset: RadiologyDataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    if config["training"].get("num_gpus", 1) > num_gpus:
        config["training"]["num_gpus"] = num_gpus

    processor = Maira2Processor.from_pretrained(
        config["model"]["name"],
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        trust_remote_code=True,
        device_map="auto",
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    if config.get("training", {}).get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(dict(use_reentrant=False))
    model = get_peft_model(model, peft_config)

    model.train()

    training_args = TrainingArguments(
        output_dir=config["model"]["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config.get("training", {}).get("gradient_accumulation_steps", 1),
        learning_rate=float(config["training"]["learning_rate"]),
        logging_steps=config["training"]["logging_steps"],
        evaluation_strategy=config["training"]["evaluation_strategy"],
        save_strategy="steps",
        save_steps=config["training"]["save_steps"],
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="wandb",
        seed=config["training"]["seed"],
        push_to_hub=False,
        dataloader_pin_memory=True,
        fp16=config.get("training", {}).get("fp16", False),
        tf32=config.get("training", {}).get("tf32", True),
        eval_steps=config["training"]["logging_steps"],
        run_name=config["wandb"]["run_name"],
        warmup_ratio=config["training"]["warmup_ratio"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        dataloader_num_workers=config.get("training", {}).get("dataloader_num_workers", os.cpu_count()),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    data_collator = partial(collate_fn, processor=processor, config=config, dataset="train")

    callbacks = [GradientNormCallback(), LearningRateLoggerCallback()]

    early_stopping_config = config["training"].get("early_stopping", {})
    if early_stopping_config.get("enabled", False):
        patience = early_stopping_config.get("patience", 3)
        early_stopping = EarlyStoppingCallback(early_stopping_patience=patience)
        callbacks.append(early_stopping)
        logger.info(f"Early stopping enabled: patience={patience}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    logger.info("Starting training...")
    torch.cuda.empty_cache()
    train_result = trainer.train()
    logger.info(f"Training completed. Final train loss: {train_result.training_loss}")

    eval_results = trainer.evaluate()
    logger.info(f"Validation loss: {eval_results.get('eval_loss', 'Not computed')}")

    trainer.save_model(config["model"]["output_dir"])
    processor.save_pretrained(config["model"]["output_dir"])


def main():
    config_path = "config.yaml"
    config = load_config(config_path)

    wandb.init(project=config["wandb"]["project_name"], entity=config["wandb"]["entity"])
    wandb.run.name = config["wandb"]["run_name"]

    random.seed(config["training"]["seed"])
    torch.manual_seed(config["training"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["training"]["seed"])

    lazy_preprocess = config["training"].get("lazy_preprocess", False)
    train_dataset, val_dataset = preprocess_data(
        config,
        subset_size=SUBSET_SIZE,
        lazy_preprocess=lazy_preprocess
    )
    train_model(config, train_dataset, val_dataset)


if __name__ == "__main__":
    main()
