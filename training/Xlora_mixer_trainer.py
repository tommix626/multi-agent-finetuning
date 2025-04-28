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

    
    def train(self):
        """Train the X-LoRA mixer."""
        print("[XLoraMixerTrainer] Starting training...")
        
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            
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
                
                # Logging
                running_loss += loss.item()
                progress_bar.set_postfix({
                    "step": step,
                    "loss": f"{loss.item():.4f}"
                })
                
                # Optional: visualize expert weights periodically
                if step % self.config.eval_steps == 0:
                    weights = self.expert_cluster.get_expert_mixing_weights()
                    if weights is not None:
                        expert_weights = weights.mean(dim=(0, 1, 2))
                        print(f"Expert mixing weights: {expert_weights}")
                
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
        
        # End of training
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
        # Create checkpoint directory
        step_suffix = f"-step-{step}" if step is not None else ""
        save_dir = self.config.base_dir
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
        
        # Save scalings log if enabled
        if self.config.log_scalings:
            try:
                self.model.flush_log_scalings(os.path.join(save_dir, "scalings_log"))
            except Exception as e:
                print(f"[XLoraMixerTrainer] Warning: Failed to save scalings log: {e}")
        
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
        
        print(f"[XLoraMixerTrainer] Saved checkpoint to {save_dir}")
    