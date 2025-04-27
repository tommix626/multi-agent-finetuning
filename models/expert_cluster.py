"""
The Expert Cluster manages the Expert Agents.
The base model is in control of the expert cluster, and the agent only provide adaptors to be load and unload (using PEFT).

"""

import random
from typing import List, Optional, Tuple

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
        print(f"[Expert cluster] deledate to expert {self.experts[chosen_expert_id].adapter.name}")
        return self.experts[chosen_expert_id]

    def _select(self, perplexities: List[float]) -> int:
        """delegation strategy for which expert to train on
    should incorporate some regularization (randomness). As per Daniel's suggestion.
        """
        index = [i for i in range(len(perplexities))]
        epsilon=0.05
        if random.random() < epsilon:
            print(f"Epsilon greedy, random pick.")
            return random.choice(index)
        perplexity_tensor = torch.tensor(perplexities)
        scores = -perplexity_tensor / self.selection_temperature
        scores = scores / torch.sum(scores)
        probs = torch.softmax(scores, dim=0).tolist()
        print(f"delegation ppl={perplexities}, scores={scores}, probs={probs}")

        self.selection_temperature = max(1,self.selection_temperature * 0.999)
        print(f"[Expert cluster] current delegation temperature {self.selection_temperature}")
        return random.choices(index, weights=probs, k=1)[0]

    def save_all_experts(self, base_save_dir):
        """save all expert's adapter"""
        for exp in self.experts:
            exp.save(base_save_dir)

    def load_all_experts(self, base_dir: Optional[str] =None):
        for exp in self.experts:
            exp.load(base_dir)
