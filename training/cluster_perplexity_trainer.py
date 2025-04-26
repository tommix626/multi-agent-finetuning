"""""
The trainer class for training the clustering phase.
Goal is to call trainer.train() to perform training.

The Trainer manages:
- Config (learning rate, schedule, hyperparameters)
- Model and tokenizer instantiation
- Optimizer setup
- Expert cluster management
- Training logic: loss calculation and backpropagation
- Callback handling for modular events and logging
"""

from dataclasses import dataclass, asdict
import json
import os
from typing import Optional, List

import torch
from torch.utils.data import DataLoader
from peft import get_peft_model
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from models.expert_cluster import ExpertCluster
from training.callbacks import Callback


@dataclass
class TrainerConfig:
    model_name: str = "EleutherAI/gpt-neo-125M"
    num_experts: int = 3
    lr: float = 1e-4
    epochs: int = 3
    device: str = "cuda"
    callbacks: Optional[List[Callback]] = None
    selection_temperature: float = 1.0
    trainer_id: str = "default-trainer"

    def expert_cluster_kwargs(self) -> dict:
        """Return arguments relevant for initializing ExpertCluster."""
        return {
            'num_experts': self.num_experts,
            'device': self.device,
            'selection_temperature': self.selection_temperature
        }


class ExpertTrainer:
    def __init__(self,
                 training_config: TrainerConfig,
                 dataloader: DataLoader,
                 peft_config):
        self.config = training_config
        self.dataloader = dataloader
        self.peft_config = peft_config

        # Model and tokenizer setup
        base_model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.peft_base_model = get_peft_model(base_model, self.peft_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Expert cluster setup
        self.expert_cluster = ExpertCluster(
            base_model=self.peft_base_model,
            tokenizer=self.tokenizer,
            peft_config=self.peft_config,
            **self.config.expert_cluster_kwargs()
        )

        # Optimizer setup
        self.optimizer = torch.optim.AdamW(
            self.peft_base_model.parameters(),
            lr=self.config.lr
        )

        # Callbacks and metrics
        self.callbacks = self.config.callbacks or []
        self.metrics = {}
        self.device = self.config.device
        self.trainer_id = self.config.trainer_id

    def train(self):
        print("[Trainer] Starting training...")
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            self._trigger_callbacks("on_epoch_start")

            running_loss = 0.0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}")
            for step, batch in enumerate(progress_bar):
                self._trigger_callbacks("on_batch_start")

                # Delegation step: select expert
                expert = self.expert_cluster.delegate_to_expert(batch)
                loss = expert.get_training_loss_on(batch)

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Logging
                running_loss += loss.item()
                progress_bar.set_postfix({
                    "step": step,
                    "expert": expert.id,
                    "loss": f"{loss.item():.4f}"
                })

                self._trigger_callbacks("on_batch_end")

            # End of epoch
            avg_loss = running_loss / len(self.dataloader)
            self.metrics['val_loss'] = avg_loss
            print(f"[Trainer] Epoch {epoch + 1} complete. Avg Loss: {avg_loss:.4f}")
            self._trigger_callbacks("on_epoch_end")

            self._save_checkpoint(epoch)

            # Early stopping check
            if any(cb.should_stop for cb in self.callbacks if hasattr(cb, 'should_stop')):
                print("[Trainer] Training halted early from callback signal.")
                break

        self._trigger_callbacks("on_train_end")

    def _save_checkpoint(self, epoch: int):
        """Save training state after an epoch."""
        # create folder for this
        base_save_dir = f"checkpoints/{self.trainer_id}-epoch-{epoch}"
        os.makedirs(base_save_dir, exist_ok=True)
        
        # Save all experts
        self.expert_cluster.save_all_experts(base_save_dir)

        # Save trainer-level metrics
        trainer_state = {
            "epoch": epoch,
            "metrics": self.metrics,
        }

        with open(os.path.join(base_save_dir, "trainer_state.json"), "w") as f:
            json.dump(trainer_state, f, indent=2)

    def _load_checkpoint(self):
        """Load training state and experts from latest available checkpoint."""
        # Find the latest checkpoint folder
        checkpoints_root = "checkpoints"
        if not os.path.exists(checkpoints_root):
            print("[Trainer] No checkpoints directory found. Starting fresh.")
            self.start_epoch = 0
            return

        # List all folders matching this trainer ID
        checkpoint_dirs = [
            d for d in os.listdir(checkpoints_root)
            if d.startswith(self.trainer_id)
        ]

        if not checkpoint_dirs:
            print("[Trainer] No matching checkpoint folders found. Starting fresh.")
            self.start_epoch = 0
            return

        # Sort checkpoints by epoch number
        def extract_epoch(folder_name: str) -> int:
            try:
                return int(folder_name.split("-epoch-")[-1])
            except Exception:
                return -1  # In case folder name is weird

        checkpoint_dirs = sorted(checkpoint_dirs, key=extract_epoch)
        latest_checkpoint = checkpoint_dirs[-1]
        latest_checkpoint_path = os.path.join(checkpoints_root, latest_checkpoint)

        print(f"[Trainer] Found checkpoint '{latest_checkpoint}'. Resuming from there.")

        # Load all experts
        self.expert_cluster.load_all_experts(latest_checkpoint_path)

        # Load trainer state
        trainer_state_path = os.path.join(latest_checkpoint_path, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, "r") as f:
                trainer_state = json.load(f)

            self.start_epoch = trainer_state.get("epoch", 0) + 1  # Resume from next epoch
            self.metrics = trainer_state.get("metrics", {})
            print(f"[Trainer] Loaded trainer state. Resuming from epoch {self.start_epoch}.")
        else:
            print("[Trainer] No trainer_state.json found. Starting from epoch 0.")
            self.start_epoch = 0

    def _trigger_callbacks(self, hook_name: str):
        for cb in self.callbacks:
            hook = getattr(cb, hook_name, None)
            if callable(hook):
                hook(self)
