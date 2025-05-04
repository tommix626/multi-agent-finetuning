from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.expert_cluster import ExpertCluster
from training.callbacks import Callback
from training.cluster_perplexity_trainer import ExpertTrainer, TrainerConfig


class TrainingBufferManager:
    """
    Manages expert-specific training buffers to accumulate and trigger batched updates.
    """
    def __init__(self, buffer_batch_size: int):
        self.buffer_batch_size = buffer_batch_size
        self.buffers: Dict[int, List[dict]] = defaultdict(list)

    def add(self, expert_id: int, batch: dict):
        """Add a batch to the expert buffer."""
        self.buffers[expert_id].append(batch)

    def is_ready(self, expert_id: int) -> bool:
        return len(self.buffers[expert_id]) >= self.buffer_batch_size

    def get_ready_batches(self, expert_id: int) -> Optional[List[dict]]:
        if self.is_ready(expert_id):
            batches = self.buffers[expert_id]
            self.buffers[expert_id] = []  # clear after consuming
            return batches
        return None

    def flush_all(self) -> Dict[int, List[dict]]:
        """Flush all remaining batches for final update at epoch end."""
        flushed = dict(self.buffers)
        self.buffers.clear()
        return flushed


class BufferedExpertTrainer(ExpertTrainer):
    def __init__(self,
                 training_config: TrainerConfig,
                 dataloader: DataLoader,
                 peft_config,
                 buffer_batch_size: int = 8):
        super().__init__(training_config, dataloader, peft_config)
        self.buffer_manager = TrainingBufferManager(buffer_batch_size)
        self.buffer_batch_size = buffer_batch_size

    def train(self):
        print("[BufferedTrainer] Starting training with buffer batching...")
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            self._trigger_callbacks("on_epoch_start")

            running_loss = 0.0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                self._trigger_callbacks("on_batch_start")

                expert_id, expert = self.expert_cluster.delegate_to_expert(batch)
                self.buffer_manager.add(expert_id, batch)

                loss = None
                if self.buffer_manager.is_ready(expert_id):
                    print(f"[Buffer Manager] Expert {expert_id} is ready! Train!")
                    ready_batches = self.buffer_manager.get_ready_batches(expert_id)
                    combined_batch = self._collate_batches(ready_batches)
                    loss = expert.get_training_loss_on(combined_batch)

                    # Optimization step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Logging
                    running_loss += loss.item()
                    progress_bar.set_postfix({
                        "step": step,
                        "expert": expert_id,
                        "loss": f"{loss.item():.4f}"
                    })

                self._trigger_callbacks("on_batch_end")

            # Final flush to make sure no data is wasted
            final_flush = self.buffer_manager.flush_all()
            for expert_id, buffered_batches in final_flush.items():
                if buffered_batches:
                    expert = self.expert_cluster.get_expert_by_id(expert_id)
                    combined_batch = self._collate_batches(buffered_batches)
                    loss = expert.get_training_loss_on(combined_batch)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()

            avg_loss = running_loss / len(self.dataloader)
            self.metrics.setdefault('val_loss', []).append(avg_loss)
            print(f"[BufferedTrainer] Epoch {epoch + 1} complete. Avg Loss: {avg_loss:.4f}")
            self._trigger_callbacks("on_epoch_end")

            self._save_checkpoint(epoch)

            if any(cb.should_stop for cb in self.callbacks if hasattr(cb, 'should_stop')):
                print("[BufferedTrainer] Training halted early from callback signal.")
                break

        self._trigger_callbacks("on_train_end")

    def _collate_batches(self, batch_list: List[dict]) -> dict:
        """
        Merge a list of individual batches (each is a dict of tensors) into a single batched dict.
        Handles flattening correctly.
        """
        merged = defaultdict(list)
        for batch in batch_list:
            for key, value in batch.items():
                merged[key].append(value)
        # Stack all tensors along batch dim
        print(f"merged={merged}")
        return {key: torch.cat(value_list, dim=0) for key, value_list in merged.items()}
