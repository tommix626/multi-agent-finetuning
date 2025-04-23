from peft import PeftConfig, PeftModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from data.datamodel import QADatapoint

from collections import defaultdict

from models.expert_adapter import ExpertAdapter

class ExpertAgent:
    def __init__(self, expert_id: int, base_peft_model: PeftModel, adapter_config: PeftConfig, tokenizer:AutoTokenizer, device:str ):
        self.id = expert_id
        self.peft_model = base_peft_model  # shared reference
        self.training_data_freq = defaultdict(int)

        self.adapter = ExpertAdapter(self.peft_model, adapter_config, f"adapter_for_expert_{self.id}", tokenizer, device)

    def compute_perplexity(self, QA_data: QADatapoint) -> float:
        """Return perplexity under this adapter."""
        raise NotImplementedError()

    def get_training_loss_on(self, QA_data: QADatapoint, lr: float):
        """Get training this expert on a single datapoint."""
        raise NotImplementedError()

    def reset_adapter(self):
        """Reset adapter weights (e.g., if overfitted)."""
        raise NotImplementedError()

    def record_datapoint(self, datapoint_id: str):
        self.training_data_freq[datapoint_id] += 1

