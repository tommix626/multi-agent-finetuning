"""
Evaluation script for ExpertCluster.
Modular, clean, and configurable via --config and --epoch.
"""

import os
import json
import argparse
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PeftConfig

from training.cluster_perplexity_trainer import TrainerConfig
from models.expert_cluster import ExpertCluster

# ---------------------- Argument and Config Loading ----------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to metadata.json or TrainerConfig YAML/JSON")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch to load (latest if None)")
    return parser.parse_args()

def find_checkpoint_folder(trainer_id: str, checkpoints_root: str, epoch: Optional[int] = None) -> str:
    checkpoint_folders = [
        d for d in os.listdir(checkpoints_root)
        if d.startswith(trainer_id)
    ]
    if not checkpoint_folders:
        raise FileNotFoundError(f"No checkpoints found for trainer_id={trainer_id}")

    checkpoint_folders = sorted(checkpoint_folders, key=lambda x: int(x.split("-epoch-")[-1]))
    if epoch is None:
        return os.path.join(checkpoints_root, checkpoint_folders[-1])
    for folder in checkpoint_folders:
        if f"-epoch-{epoch}" in folder:
            return os.path.join(checkpoints_root, folder)
    raise FileNotFoundError(f"No checkpoint found for epoch {epoch}")

def load_training_config(config_path: str) -> TrainerConfig:
    if config_path.endswith("metadata.json"):
        with open(config_path, "r") as f:
            metadata = json.load(f)
        return TrainerConfig(**metadata["training_config"])
    raise ValueError("Only metadata.json loading is supported in this script.")

# ---------------------- Model and Cluster Loading ----------------------

def load_cluster_and_tokenizer(config: TrainerConfig, checkpoint_path: str) -> tuple[ExpertCluster, AutoTokenizer]:
    base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    peft_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.exists(peft_config_path):
        raise FileNotFoundError(f"Missing PEFT config at {peft_config_path}")
    peft_config = PeftConfig.from_pretrained(checkpoint_path)

    peft_model = get_peft_model(base_model, peft_config).to(config.device)

    cluster = ExpertCluster(
        base_model=peft_model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        **config.expert_cluster_kwargs()
    )
    cluster.load_all_experts(checkpoint_path)
    return cluster, tokenizer

# ---------------------- Evaluation Logic ----------------------

def evaluate_model(model, batch, device):
    model.eval()
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        perplexity = torch.exp(loss)

        return {"loss": loss.item(), "perplexity": perplexity.item(), "accuracy": 0.0}  # Add real acc if needed

# ---------------------- Main Script ----------------------

def main():
    args = parse_args()
    config = load_training_config(args.config)
    checkpoint_path = find_checkpoint_folder(config.trainer_id, "checkpoints", epoch=args.epoch)
    print(f"[Eval] Loading checkpoint from {checkpoint_path}")

    cluster, tokenizer = load_cluster_and_tokenizer(config, checkpoint_path)

    from your_dataset_module import get_eval_dataset  # You must define this
    eval_dataset = get_eval_dataset(tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False)

    total_metrics = {"loss": 0.0, "perplexity": 0.0, "accuracy": 0.0}
    total_batches = 0

    for batch in eval_loader:
        expert = cluster.delegate_to_expert(batch)
        batch_metrics = evaluate_model(expert.adapter.model, batch, config.device)

        for k in total_metrics:
            total_metrics[k] += batch_metrics[k]
        total_batches += 1

    avg_metrics = {k: v / total_batches for k, v in total_metrics.items()}

    print("\n=== Evaluation Results ===")
    for k, v in avg_metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

if __name__ == "__main__":
    main()

