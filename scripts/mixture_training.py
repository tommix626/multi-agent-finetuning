"""
Mixer training script for the X-LoRA mixture-of-experts model.

This script assumes expert adapters have already been trained and saved via `cluster_training.py`.
It will:
  1. Load config from YAML/JSON
  2. Initialize data loaders
  3. Instantiate an ExpertCluster and load trained adapters
  4. Convert the cluster into an X-LoRA model
  5. Train only the mixer scalings, freezing base model and adapters
"""
import argparse
import json
import os
from pathlib import Path
from dataclasses import asdict

import yaml
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from data.mmlu.mmludataset import pre_process_data
from models.expert_cluster import ExpertCluster
from training.Xlora_mixer_trainer import XLoraMixerConfig, XLoraMixerTrainer
from training.callbacks import ModelCheckpoint, EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the X-LoRA mixer over pretrained expert adapters."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML or JSON config file."
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    if path.endswith(('.yaml', '.yml')):
        with open(path) as f:
            return yaml.safe_load(f)
    if path.endswith('.json'):
        with open(path) as f:
            return json.load(f)
    raise ValueError("Config must be .yaml/.yml or .json")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Hyperparameters
    model_name = cfg.get("model_name", "EleutherAI/gpt-neo-125M")
    train_bs = cfg.get("batch_size", 16)
    eval_bs = cfg.get("eval_batch_size", train_bs)
    num_experts = cfg.get("num_experts", 4)
    lr = cfg.get("learning_rate", 1e-4)
    epochs = cfg.get("num_epochs", 3)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    cluster_ckpt = cfg["cluster_checkpoint_dir"]  # required
    save_dir = cfg.get("save_dir", "./checkpoints")
    trainer_id = cfg.get("trainer_id", "xlora-mixer")

    # LoRA config (must match cluster training)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg.get("lora_r", 8),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.1),
        bias="none",
        target_modules=cfg.get("target_modules", ["q_proj", "v_proj"])
    )

    # 1. Prepare data loaders
    print("[Mixer] Loading datasets...")
    train_loader, eval_loader, _ = pre_process_data(
        model_name=model_name,
        batch_size=train_bs,
        device=device,
        peft_config=peft_config
    )

    # 2. Instantiate PEFT base model and tokenizer
    print("[Mixer] Loading base model and adapters...")
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    peft_base = get_peft_model(base_model, peft_config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Build cluster and load trained adapters
    cluster = ExpertCluster(
        base_model=peft_base,
        tokenizer=tokenizer,
        peft_config=peft_config,
        num_experts=num_experts,
        device=device,
        selection_temperature=cfg.get("selection_temperature", 1.0)
    )
    cluster.load_all_experts(cluster_ckpt)

    # 4. Mixer training config
    mixer_cfg = XLoraMixerConfig(
        lr=lr,
        epochs=epochs,
        batch_size=train_bs,
        save_steps=cfg.get("save_steps", 100),
        eval_steps=cfg.get("eval_steps", 50),
        log_scalings=cfg.get("log_scalings", True),
        xlora_depth=cfg.get("xlora_depth", 8),
        layerwise_scalings=cfg.get("layerwise_scalings", True),
        softmax_temperature=cfg.get("softmax_temperature", 1.0),
        top_k_lora=cfg.get("top_k_lora", None),
        trainer_id=trainer_id
    )

    # 5. Callbacks
    callbacks = [
        ModelCheckpoint(save_dir=save_dir, monitor="eval_loss", mode="min"),
        EarlyStopping(
            patience=cfg.get("early_stopping_patience", 2),
            monitor="eval_loss", mode="min"
        )
    ]

    # 6. Instantiate and run trainer
    mixer_trainer = XLoraMixerTrainer(
        config=mixer_cfg,
        expert_cluster=cluster,
        dataloader=train_loader,
        eval_dataloader=eval_loader,
        callbacks=callbacks
    )
    mixer_trainer.train()


if __name__ == "__main__":
    main()
