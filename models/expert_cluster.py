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
import xlora
from transformers import AutoConfig
import os
import json

class ExpertCluster:
    def __init__(self, base_model: PeftModel, tokenizer:AutoTokenizer, peft_config: PeftConfig, num_experts: int, device: str, selection_temperature:float):
        self.peft_config = peft_config
        self.peft_model, self.tokenizer = base_model, tokenizer
        # self.peft_model, eself.tokenizer = self._setup_peft_model(base_model_name)
        self.device = device
        self.base_model_name = base_model.base_model.name_or_path
        print("base model name: ", self.base_model_name)

        self.experts: List[ExpertAgent] = []
        self._init_agent(num_experts)

        self.selection_temperature = selection_temperature

        self.mixer_trained = False
        self.xlora_model = None

        self.base_dir = ""


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
        scores = -perplexity_tensor
        scores = scores / torch.sum(scores) * 5/self.selection_temperature # normalize to sum to 5/temp
        probs = torch.softmax(scores, dim=0).tolist()
        print(f"delegation ppl={perplexities}, scores={scores}, probs={probs}")

        self.selection_temperature = max(0.1,self.selection_temperature * 0.998)
        print(f"[Expert cluster] current delegation temperature {self.selection_temperature}")
        return random.choices(index, weights=probs, k=1)[0]

    def save_all_experts(self, base_save_dir):
        """save all expert's adapter"""
        self.peft_model.save_pretrained(base_save_dir)

        # also save the frequency
        for exp in self.experts:
            exp.save(base_save_dir)

    def load_all_experts(self, base_dir: Optional[str] =None):
        for exp in self.experts:
            exp.load(base_dir)
        self.base_dir = base_dir

    def convert_to_xlora(self, xlora_depth: int = 8, enable_softmax: bool = True, 
                         softmax_temperature: float = 1.0, layerwise_scalings: bool = True):
        """
        Convert the model to use X-LoRA for mixing experts.
        This creates a new xlora_model without modifying the existing peft_model.
        """

        if self.xlora_model is not None:
            print("[Expert Cluster] Model has already been converted to X-LoRA.")
            return
            
        # Get paths to all expert adapters
        adapter_paths = {}
        for expert in self.experts:
            adapter_name = expert.adapter.name
            adapter_path = os.path.join(self.base_dir, expert.adapter.name)
            adapter_paths[adapter_name] = adapter_path
            print(f"[Expert Cluster] Adding adapter {adapter_name} from {adapter_path}")
        
        # Convert model to X-LoRA
        print(f"[Expert Cluster] Converting model to X-LoRA with {len(adapter_paths)} experts")
        model_hidden_size = self.peft_model.base_model.config.hidden_size
        self.xlora_model = xlora.add_xlora_to_model(
            model=self.peft_model,
            xlora_config=xlora.xLoRAConfig(
                hidden_size=model_hidden_size,
                base_model_id=self.base_model_name,
                xlora_depth=xlora_depth,
                device=torch.device("cuda"),
                adapters=adapter_paths,
                enable_softmax=enable_softmax,
                softmax_temperature=softmax_temperature,
                layerwise_scalings=layerwise_scalings,
                # Freeze all adapters and only train the mixer
                use_trainable_adapters=False
            ),
            verbose=True
        )
        self.xlora_model = self.xlora_model.to(torch.device(self.device))
        print("[Expert Cluster] Successfully converted model to X-LoRA")
        print(f"[Expert Cluster] x-lora model device: {next(self.xlora_model.parameters()).device}")
    
    def get_xlora_model(self):
        """Get the X-LoRA model if available."""
        if self.xlora_model is None:
            raise ValueError("Model has not been converted to X-LoRA. Call convert_to_xlora() first.")
        return self.xlora_model
    
    def save_xlora_model(self, save_path: str):
        """Save the X-LoRA model."""
        if self.xlora_model is None:
            raise ValueError("Model has not been converted to X-LoRA. Nothing to save.")
        
        os.makedirs(save_path, exist_ok=True)
        self.xlora_model.save_pretrained(save_path)
        print(f"[Expert Cluster] Saved X-LoRA model to {save_path}")
        
        # Also save a marker that this model has been mixer trained
        self.mixer_trained = True
        with open(os.path.join(save_path, "mixer_trained.json"), "w") as f:
            json.dump({"mixer_trained": True}, f)
    
    def load_xlora_model(self, load_path: str):
        """Load an X-LoRA model."""
        
        print(f"[Expert Cluster] Loading X-LoRA model from {load_path}")
        self.xlora_model = xlora.from_pretrained(
            load_path,
            self.peft_model,
            self.device,
        )
        self.xlora_model = self.xlora_model.to(torch.device(self.device))

        # Check if model was mixer trained
        mixer_trained_path = os.path.join(load_path, "mixer_trained.json")
        if os.path.exists(mixer_trained_path):
            with open(mixer_trained_path, "r") as f:
                data = json.load(f)
                self.mixer_trained = data.get("mixer_trained", False)
        
        print(f"[Expert Cluster] Successfully loaded X-LoRA model (mixer_trained={self.mixer_trained})")
    
    def enable_xlora_logging(self):
        """Enable logging of expert mixing weights."""
        if self.xlora_model is None:
            raise ValueError("Model has not been converted to X-LoRA.")
        self.xlora_model.enable_scalings_logging()
        print("[Expert Cluster] Enabled X-LoRA scalings logging")
    
    def get_expert_mixing_weights(self):
        """Get the latest expert mixing weights."""
        if self.xlora_model is None:
            raise ValueError("Model has not been converted to X-LoRA.")
        return self.xlora_model.get_latest_scalings()    
    
