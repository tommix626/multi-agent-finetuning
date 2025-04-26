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

            # Early stopping check
            if any(cb.should_stop for cb in self.callbacks if hasattr(cb, 'should_stop')):
                print("[Trainer] Training halted early from callback signal.")
                break

        self._trigger_callbacks("on_train_end")
        self.expert_cluster.save_all_experts()

    def _trigger_callbacks(self, hook_name: str):
        for cb in self.callbacks:
            hook = getattr(cb, hook_name, None)
            if callable(hook):
                hook(self)
