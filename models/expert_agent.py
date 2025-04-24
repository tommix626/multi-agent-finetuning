from peft import PeftConfig, PeftModel
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from data.datamodel import QADatapoint

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
            labels = batch['input_ids'].clone().to(self.device)

            outputs = self.peft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        return perplexity



    def get_training_loss_on(self, QA_data: QADatapoint, lr: float, record_training=True):
        """Get training this expert on a single datapoint."""
        #TODO: Activate the adapter, forward pass, then query the log prob. Follow Hw6/Hw7
        raise NotImplementedError()

    def reset_adapter(self):
        """Reset adapter weights (e.g., if overfitted)."""
        # fetch the current setting
        adapter_config = self.adapter.peft_config
        tokenizer = self.adapter.tokenizer
        device = self.adapter.device

        # restart an adapter from scratch
        self.adapter = ExpertAdapter(self.peft_model, adapter_config, f"adapter_for_expert_{self.id}", tokenizer, device, force_new=True)

    def record_datapoint(self, datapoint_id: str):
        self.training_data_freq[datapoint_id] += 1

