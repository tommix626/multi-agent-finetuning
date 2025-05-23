#!/usr/bin/env python3
"""
XLoraMixerTrainer: Specialized trainer for the X-LoRA mixture-of-experts model.
This trainer freezes the base model and LoRA adapters and only trains the X-LoRA mixer.
"""

import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.expert_cluster import ExpertCluster


@dataclass
class XLoraMixerConfig:
    """Configuration for training the X-LoRA mixer."""
    lr: float = 1e-4
    epochs: int = 3
    batch_size: int = 16
    save_steps: int = 100
    eval_steps: int = 50
    log_scalings: bool = True
    xlora_depth: int = 8
    layerwise_scalings: bool = True
    softmax_temperature: float = 1.0
    top_k_lora: Optional[int] = None
    base_dir: str = None


class XLoraMixerTrainer:
    """
    Trainer for the X-LoRA mixer that learns how to combine expert adapters.
    This trainer assumes the expert cluster has already been converted to use X-LoRA.
    """
    
    def __init__(
        self,
        config: XLoraMixerConfig,
        expert_cluster: ExpertCluster,
        dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.expert_cluster = expert_cluster
        self.dataloader = dataloader
        self.eval_dataloader = eval_dataloader
        
        # Ensure expert cluster has been converted to X-LoRA
        if not hasattr(expert_cluster, 'xlora_model') or expert_cluster.xlora_model is None:
            print("[XLoraMixerTrainer] Converting expert cluster to X-LoRA...")
            expert_cluster.convert_to_xlora(
                xlora_depth=config.xlora_depth,
                layerwise_scalings=config.layerwise_scalings,
                softmax_temperature=config.softmax_temperature
            )
        
        self.model = expert_cluster.get_xlora_model()
        self.model.to(self.expert_cluster.device)

        # Set top-k if specified
        if config.top_k_lora is not None:
            self.model.set_topk_lora(config.top_k_lora)
            print(f"[XLoraMixerTrainer] Set top-k LoRA to {config.top_k_lora}")
        
        # Count trainable parameters
        self.model.print_trainable_parameters()
        
        # Set up optimizer - only train X-LoRA classifier parameters
        # The base model and all adapters remain frozen
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr
        )
        
        # Metrics tracking
        self.metrics = {
            "train_loss": [],
            "eval_loss": []
        }
        
        # Enable logging if requested
        if config.log_scalings:
            self.expert_cluster.enable_xlora_logging()

        self.debug_trainable_parameters()
        
    def debug_trainable_parameters(self):
        """
        Debug which parameters are trainable in the model.
        Prints parameter groups and their trainable status.
        """
        print("\n======= DEBUGGING TRAINABLE PARAMETERS =======")
        
        # Track parameters by type
        param_stats = {
            "base_model": {"trainable": 0, "frozen": 0},
            "lora_adapters": {"trainable": 0, "frozen": 0},
            "xlora_mixer": {"trainable": 0, "frozen": 0},
            "other": {"trainable": 0, "frozen": 0}
        }
        
        # Counter for trainable parameter groups
        trainable_groups = []
        
        # Inspect all named parameters
        for name, param in self.model.named_parameters():
            param_type = "other"
            
            # Categorize the parameter
            if "xlora" in name.lower() and "adapter" not in name.lower():
                param_type = "xlora_mixer"
            elif "lora" in name.lower() or "adapter" in name.lower():
                param_type = "lora_adapters"
            elif "base_model" in name.lower():
                param_type = "base_model"
            
            # Count trainable vs frozen parameters
            if param.requires_grad:
                param_stats[param_type]["trainable"] += param.numel()
                trainable_groups.append((name, param.shape, param.numel()))
            else:
                param_stats[param_type]["frozen"] += param.numel()
        
        # Print summary
        print("Parameter statistics by component:")
        for component, stats in param_stats.items():
            total = stats["trainable"] + stats["frozen"]
            if total > 0:
                trainable_percent = 100 * stats["trainable"] / total
                print(f"  {component}: {stats['trainable']:,}/{total:,} params trainable ({trainable_percent:.2f}%)")
        
        # Print detailed list of trainable parameter groups
        print("\nTrainable parameter groups:")
        trainable_groups.sort(key=lambda x: x[0])  # Sort by name
        for name, shape, count in trainable_groups:
            print(f"  {name}: shape={shape}, params={count:,}")
        
        print("============================================\n")
    
    def train(self):
        """Train the X-LoRA mixer."""
        print("[XLoraMixerTrainer] Starting training...")

        # --- ADDED: prepare loss log file ---
        loss_log_path = "/home/rliu79/test/multi-agent-finetuning/logs/outputs/loss.txt"
        os.makedirs(os.path.dirname(loss_log_path), exist_ok=True)
        loss_file = open(loss_log_path, "a")
        # ---------------------------------------

        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            
            # Print trainable parameters at the start of each epoch
            print("[DEBUG] Trainable parameters at start of epoch:")
            for name, param in self.model.named_parameters():
                status = "requires_grad=True" if param.requires_grad else "frozen"
                print(f"  {name}: {status}")

            running_loss = 0.0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                
                # Forward pass through the X-LoRA model
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.expert_cluster.device),
                    attention_mask=batch["attention_mask"].to(self.expert_cluster.device),
                    labels=batch["labels"].to(self.expert_cluster.device)
                )
                
                loss = outputs.loss
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # --- ADDED: write to loss log file ---
                loss_file.write(f"{epoch+1} - {step} - {loss.item():.4f}\n")
                # ----------------------------------------

                # Logging
                running_loss += loss.item()
                progress_bar.set_postfix({
                    "step": step,
                    "loss": f"{loss.item():.4f}"
                })
                
                # Save checkpoint if needed
                if step > 0 and step % self.config.save_steps == 0:
                    self._save_checkpoint(epoch, step)
            
            # End of epoch
            avg_loss = running_loss / len(self.dataloader)
            self.metrics["train_loss"].append(avg_loss)
            print(f"[XLoraMixerTrainer] Epoch {epoch + 1} complete. Avg Loss: {avg_loss:.4f}")
            
            # Evaluate if evaluation dataloader is available
            if self.eval_dataloader is not None:
                eval_loss = self._evaluate()
                self.metrics["eval_loss"].append(eval_loss)
                print(f"[XLoraMixerTrainer] Evaluation Loss: {eval_loss:.4f}")
            
            # Save checkpoint at end of epoch
            self._save_checkpoint(epoch)

        # --- ADDED: close loss log file ---
        loss_file.close()
        # ---------------------------------------

        print("[XLoraMixerTrainer] Training complete!")
    
    def _evaluate(self):
        """Evaluate the model on the evaluation dataloader."""
        print("[XLoraMixerTrainer] Evaluating model...")
        self.model.eval()
        
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.expert_cluster.device),
                    attention_mask=batch["attention_mask"].to(self.expert_cluster.device),
                    labels=batch["labels"].to(self.expert_cluster.device)
                )
                
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(self.eval_dataloader)
        self.model.train()
        return avg_loss
    
    def _save_checkpoint(self, epoch: int, step: Optional[int] = None):
        """Save the X-LoRA model and training state."""
        # Build mixer-specific checkpoint path:
        step_suffix = f"-step-{step}" if step is not None else ""
        base = self.config.base_dir.rstrip("/")
        save_dir = f"{base}-mixer{step_suffix}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the X-LoRA model
        self.expert_cluster.save_xlora_model(save_dir)
        
        # Save trainer state
        trainer_state = {
            "epoch": epoch,
            "step": step,
            "metrics": self.metrics,
            "config": asdict(self.config)
        }
        
        with open(os.path.join(save_dir, "trainer_state.json"), "w") as f:
            json.dump(trainer_state, f, indent=2)
        
        # Save metadata
        metadata = {
            "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "xlora_depth": self.config.xlora_depth,
            "layerwise_scalings": self.config.layerwise_scalings,
            "softmax_temperature": self.config.softmax_temperature,
            "top_k_lora": self.config.top_k_lora
        }
        
        with open(os.path.join(save_dir, "xlora_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[XLoraMixerTrainer] Saved mixer checkpoint to {save_dir}")
