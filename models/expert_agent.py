import json
import os
import random
from typing import Optional
from peft import PeftConfig, PeftModel
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
import torch.nn.functional as F  # needed for softmax

from collections import defaultdict

from models.expert_adapter import ExpertAdapter

class ExpertAgent:
    def __init__(self, expert_id: int, base_peft_model: PeftModel, adapter_config: PeftConfig, tokenizer:AutoTokenizer, device:str ):
        self.id = expert_id
        self.peft_model = base_peft_model  # shared reference
        self.training_data_freq = defaultdict(int)
        self.device = device
        self.tokenizer = tokenizer

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
        if random.random() < 0.02:
            # Verbose debugging
            print("[DEBUG] ===== MODEL FORWARD =====")
            print(f"[DEBUG] Input IDs: {input_ids[0].tolist()}")
            print(f"[DEBUG] Labels:    {labels[0].tolist()}")

            # Convert logits to predicted token ids (greedy argmax)
            logits = outputs.logits
            pred_ids = logits.argmax(dim=-1)
            print(f"[DEBUG] Predicted: {pred_ids[0].tolist()}")
            
            # Show token-level loss if needed
            if hasattr(outputs, "loss") and outputs.loss is not None:
                print(f"[DEBUG] Loss: {outputs.loss.item()}")

                
            # Optionally decode if tokenizer is available
            if hasattr(self, "tokenizer"):
                decoded_input = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                decoded_label = self.tokenizer.decode(labels[0][labels[0] != -100], skip_special_tokens=True)
                decoded_pred = self.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
                print(f"[DEBUG] Decoded Input:  {decoded_input}")
                print(f"[DEBUG] Decoded Label:  {decoded_label}")
                print(f"[DEBUG] Decoded Output: {decoded_pred}")

                # Print top-20 tokens with probabilities at each label position
                probs = F.softmax(logits[0], dim=-1)  # (seq_len, vocab_size)
                print("[DEBUG] Top-20 token predictions per position:")
                for i, (logit_vec, label_id) in enumerate(zip(logits[0], labels[0])):
                    if label_id == -100:
                        continue  # skip padding/untrained positions
                    top_probs, top_ids = torch.topk(probs[i], 20)
                    tokens = self.tokenizer.convert_ids_to_tokens(top_ids.tolist())
                    print(f"  Position {i}: Label={self.tokenizer.convert_ids_to_tokens([label_id.item()])[0]}")
                    for tok, prob in zip(tokens, top_probs.tolist()):
                        print(f"    {tok:12s} : {prob:.4f}")


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
