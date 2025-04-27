"""
Training script for Expert Clustering with PEFT.
Supports loading hyperparameters from YAML or JSON config files.
"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
import torch
import yaml

from peft import LoraConfig, TaskType
from training.callbacks import ModelCheckpoint, EarlyStopping
from data.mmlu.mmludataset import pre_process, pre_process_data
from training.cluster_perplexity_trainer import TrainerConfig, ExpertTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Expert Cluster Model with PEFT.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML or JSON config file.")
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif config_path.endswith(".json"):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError("Config file must be a .yaml, .yml, or .json file.")
    return config

def main():
    args = parse_args()
    config_dict = load_config(args.config)

    # 1. Load hyperparameters
    model_name = config_dict.get("model_name", "EleutherAI/gpt-neo-125M")
    batch_size = config_dict.get("batch_size", 8)
    num_epochs = config_dict.get("num_epochs", 3)
    num_experts = config_dict.get("num_experts", 4)
    learning_rate = config_dict.get("learning_rate", 1e-4)
    device = config_dict.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    save_dir = config_dict.get("save_dir", "./checkpoints")
    trainer_id = config_dict.get("trainer_id", "default-trainer")

    # 2. Define LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config_dict.get("lora_r", 8),
        lora_alpha=config_dict.get("lora_alpha", 32),
        lora_dropout=config_dict.get("lora_dropout", 0.1),
        bias="none",
        target_modules=["q_proj", "v_proj"]
    )

    # 3. Prepare dataset and base model with PEFT
    print("[Main] Loading data and model...")
    train_loader, _, _ = pre_process_data(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        peft_config=peft_config
    )

    # 4. Setup callbacks
    callbacks = [
        ModelCheckpoint(save_dir=save_dir, monitor="val_loss", mode="min"),
        EarlyStopping(patience=config_dict.get("early_stopping_patience", 2), monitor="val_loss", mode="min")
    ]

    # 5. Trainer configuration
    trainer_config = TrainerConfig(
        model_name=model_name,
        num_experts=num_experts,
        lr=learning_rate,
        epochs=num_epochs,
        device=device,
        callbacks=callbacks,
        selection_temperature=config_dict.get("selection_temperature", 1.0),
        trainer_id=trainer_id,
    )

    # 6. Instantiate trainer
    trainer = ExpertTrainer(
        training_config=trainer_config,
        dataloader=train_loader,
        peft_config=peft_config
    )

    # 7. Start training
    trainer.train()

if __name__ == "__main__":
    main()
