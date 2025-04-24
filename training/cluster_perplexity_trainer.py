"""
The trainer class for training the clustering phase.
goal is to just call trainer.train() to perform training.

The trainer should be managing:
- config (lr, schedule, hyperparam)
- model and tokenizer, whose name should be included in the config, and instantiate here.
- optimizer.
- instantiate an agent cluster.
- get the loss and backprop logic. 
- using callback and logging to enhance modularity and logic.
"""
from peft import get_peft_model
import torch
from typing import Optional, List

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.utils.dummy_pt_objects import AutoModelForCausalLM
from models.expert_cluster import ExpertCluster
from models.expert_agent import ExpertAgent
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.callbacks import Callback


class TrainerConfig:
    def __init__(self,
                 model_name: str = "EleutherAI/pythia-70m",
                 num_experts: int = 3,
                 lr: float = 1e-4,
                 epochs: int = 3,
                 device: str = "cuda",
                 callbacks: Optional[List[Callback]] = None):

        self.model_name = model_name
        self.num_experts = num_experts
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.callbacks = callbacks


class ExpertTrainer:
    def __init__(self,
                 training_config: TrainerConfig,
                 dataloader: DataLoader,
                 peft_config):
        self.config = training_config
        self.dataloader = dataloader
        self.peft_config = peft_config
        base_model = AutoModelForCausalLM.from_pretrained(training_config.model_name)
        self.peft_base_model = get_peft_model(base_model, peft_config,)
        self.tokenizer = AutoTokenizer.from_pretrained(training_config.model_name)

        self.expert_cluster = ExpertCluster(
            base_model=self.peft_base_model,
            tokenizer=self.tokenizer,
            peft_config=peft_config,
            num_experts=training_config.num_experts,
            device=training_config.device
        )
        self.optimizer = torch.optim.AdamW(
            self.peft_base_model.parameters(), lr=training_config.lr
        )

        self.callbacks = training_config.callbacks or []
        self.metrics = {}  # useful for val_loss, early stopping etc.
        self.device = training_config.device

    def train(self):
        print("[Trainer] Starting training...")
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            self._trigger_callbacks("on_epoch_start")

            running_loss = 0.0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")
            for step, batch in enumerate(progress_bar):
                self._trigger_callbacks("on_batch_start")

                # Delegation step
                expert = self.expert_cluster.delegate_to_expert(batch)
                loss = expert.get_training_loss_on(batch)

                # Backward & optimizer
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix({
                    "step": step,
                    "expert": expert.id,
                    "loss": f"{loss.item():.4f}"
                })

                self._trigger_callbacks("on_batch_end")

            avg_loss = running_loss / len(self.dataloader)
            self.metrics['val_loss'] = avg_loss
            print(f"[Trainer] Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")
            self._trigger_callbacks("on_epoch_end")

            if any(cb.should_stop for cb in self.callbacks if hasattr(cb, 'should_stop')):
                print("[Trainer] Training halted early from callback signal.")
                break

        self._trigger_callbacks("on_train_end")

    def _trigger_callbacks(self, hook_name: str):
        for cb in self.callbacks:
            hook = getattr(cb, hook_name, None)
            if callable(hook):
                hook(self)
