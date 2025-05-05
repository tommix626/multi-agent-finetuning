import json
import os
from typing import List, Optional
from peft import PeftConfig, PeftModel
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer

from collections import defaultdict

from models.expert_adapter import ExpertAdapter
import torch.nn.functional as F

class ExpertAgent:
    def __init__(self, expert_id: int, base_peft_model: PeftModel, adapter_config: PeftConfig, tokenizer:AutoTokenizer, device:str ):
        self.id = expert_id
        self.peft_model = base_peft_model  # shared reference
        self.training_data_freq = defaultdict(int)
        self.device = device

        self.adapter = ExpertAdapter(self.peft_model, adapter_config, f"adapter_for_expert_{self.id}", tokenizer, device)


    def compute_perplexity(self, batch: dict) -> List[float]:
        """
        Compute per-sample perplexities for a batch under this expert.
        Returns a list of length batch_size.
        """
        self.adapter.activate()
        self.peft_model.eval()

        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.peft_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()

            # Flatten for loss computation
            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            ).view(shift_labels.size())

            # Masked mean loss per sample
            loss_per_sample = (loss_per_token * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)

            perplexity_per_sample = torch.exp(loss_per_sample).tolist()
            return perplexity_per_sample



    def get_training_loss_on(self, batch: dict, record_training=True):
        """Get training this expert on a single datapoint."""
        # Activate the adapter, forward pass, then query the log prob. Follow Hw6/Hw7
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"[DEBUG] {k}.shape = {v.shape}")
        self.adapter.activate()
        self.peft_model.train()

        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].clone().to(self.device)

        outputs = self.peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        self.record_datapoint(batch)
        
        return loss

    def reset_adapter(self):
        """Reset adapter weights (e.g., if overfitted)."""
        # fetch the current setting
        adapter_config = self.adapter.peft_config
        tokenizer = self.adapter.tokenizer
        device = self.adapter.device

        # restart an adapter from scratch
        self.adapter = ExpertAdapter(self.peft_model, adapter_config, f"adapter_for_expert_{self.id}", tokenizer, device, force_new=True)

    def record_datapoint(self, batch: dict):
        for uid in batch["uid"]:
            datapoint_id = uid
            self.training_data_freq[datapoint_id] += 1

    def save(self, base_save_dir: Optional[str]=None):
        """Save the adapter and the training data statistics."""
        agent_save_dir = self.adapter.get_save_dir() if base_save_dir is None else os.path.join(base_save_dir, self.adapter.name)
        os.makedirs(agent_save_dir, exist_ok=True)

        # self.adapter.save_adapter(agent_save_dir)

        # Save training data frequency
        freq_path = os.path.join(agent_save_dir, "training_data_freq.json")


        freq_dict = dict(self.training_data_freq)
        with open(freq_path, "w") as f:
            json.dump(freq_dict, f, indent=2)

        if self.adapter.verbose:
            print(f"[ExpertAgent] Saved training data frequency to {freq_path}.")

    def load(self, base_save_dir: Optional[str] = None):
        """Load the adapter and the training data statistics."""
        # Compute the agent's save directory
        agent_save_dir = self.adapter.get_save_dir() if base_save_dir is None else os.path.join(base_save_dir, self.adapter.name)

        # Load adapter
        self.adapter.load(agent_save_dir)

        # Load training data frequency
        freq_path = os.path.join(agent_save_dir, "training_data_freq.json")

        if os.path.isfile(freq_path):
            with open(freq_path, "r") as f:
                freq_dict = json.load(f)
            self.training_data_freq = defaultdict(int, freq_dict)
            if self.adapter.verbose:
                print(f"[ExpertAgent] Loaded training data frequency from {freq_path}.")
        else:
            if self.adapter.verbose:
                print(f"[ExpertAgent] No training data frequency file found at {freq_path}. Starting fresh.")
            self.training_data_freq = defaultdict(int)
