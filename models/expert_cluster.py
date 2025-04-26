"""
The Expert Cluster manages the Expert Agents.
The base model is in control of the expert cluster, and the agent only provide adaptors to be load and unload (using PEFT).

"""

import random
from typing import List, Tuple

from peft import PeftConfig, PeftModel, peft_model
import torch
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from models.expert_adapter import ExpertAdapter
from models.expert_agent import ExpertAgent
import transformers

class ExpertCluster:
    def __init__(self, base_model: PeftModel, tokenizer:AutoTokenizer, peft_config: PeftConfig, num_experts: int, device: str, selection_temperature:float):
        self.peft_config = peft_config
        self.peft_model, self.tokenizer = base_model, tokenizer
        # self.peft_model, self.tokenizer = self._setup_peft_model(base_model_name)
        self.device = device

        self.experts: List[ExpertAgent] = []
        self._init_agent(num_experts)

        self.selection_temperature = selection_temperature



    def _setup_peft_model(self, base_model_name: str) -> Tuple[PeftModel, AutoTokenizer]:
        """setup the base model and return the model."""
        base_model= AutoModelForCausalLM.from_pretrained(base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        return PeftModel(base_model, self.peft_config), tokenizer


    def _init_agent(self, num_agents:int):
        # init experts
        for i in range(num_agents):
            self.experts.append(ExpertAgent(i, self.peft_model, self.peft_config, self.tokenizer, self.device))  # expert themselves manage the adapter logics.


    def delegate_to_expert(self, batch:dict) -> ExpertAgent:
        perplexities = [exp.compute_perplexity(batch) for exp in self.experts]
        chosen_expert_id = self._select(perplexities)
        return self.experts[chosen_expert_id]

    def _select(self, perplexities: List[float]) -> int:
        """delegation strategy for which expert to train on
    should incorporate some regularization (randomness). As per Daniel's suggestion.
        """
        perplexities = [-p for p in perplexities]
        perplexity_tensor = torch.tensor(perplexities)
        scores = -perplexity_tensor / self.selection_temperature
        index = [i for (i,_) in enumerate(perplexities)]
        return random.choices(index, perplexities)[0]

