import json
import os
from typing import Optional
from peft import PeftConfig, PeftModel
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer

from collections import defaultdict

from models.expert_adapter import ExpertAdapter

class ExpertAgent:
    def __init__(self, expert_id: int, base_peft_model: PeftModel, adapter_config: PeftConfig, tokenizer:AutoTokenizer, device:str ):
        self.id = expert_id
        self.peft_model = base_peft_model  # shared reference
        self.training_data_freq = defaultdict(int)
        self.device = device

        self.adapter = ExpertAdapter(self.peft_model, adapter_config, f"adapter_for_expert_{self.id}", tokenizer, device)

    def compute_perplexity(self, batch: dict) -> float:
        """Return perplexity under this adapter."""
        self.adapter.activate()
        self.peft_model.eval()
        
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].clone().to(self.device)

            outputs = self.peft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        return perplexity



    def get_training_loss_on(self, batch: dict, record_training=True):
        """Get training this expert on a single datapoint."""
        # Activate the adapter, forward pass, then query the log prob. Follow Hw6/Hw7
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
        agent_save_dir = self.adapter.get_save_dir() if base_save_dir is None else os.path.join(base_save_dir, self.adapter.get_relative_save_dir())
        os.makedirs(agent_save_dir, exist_ok=True)

        self.adapter.save_adapter(agent_save_dir)

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
        agent_save_dir = self.adapter.get_save_dir() if base_save_dir is None else os.path.join(base_save_dir, self.adapter.get_relative_save_dir())

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
