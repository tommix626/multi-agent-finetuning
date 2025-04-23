"""
The LoRA Adapter class for base model finetuning
Extract all the loading/unloading/saving of the adapter logic out from ExpertAgent.
"""
from typing import Optional
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers.integrations.peft import PeftAdapterMixin
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
import torch
import os

class ExpertAdapter:
    def __init__(self, base_peft_model: PeftModel, config:PeftConfig, name: str, tokenizer, device="cuda", target_modules=None):
        self.device = device
        self.base_peft_model = base_peft_model
        self.tokenizer = tokenizer
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.name = name

    def _path_to_file(self, version=1, path_only:bool=False)-> str:
        path= f"../checkpoints/"
        if path_only:
            return path

        if version == 1:
            return path + f"{self.name}"

    def attach_raw_to_base_model(self, lora_config: PeftConfig):
        """
        Attach a itself (a fresh new LoRA adapter) to the base model for training on this example.
        """
        self.base_peft_model.add_adapter(lora_config, self.name)
        print(f"[ExpertAdapter] Adding adapter {self.name} to base model.")

    def load_adapter(self, adapter_path: Optional[str]=None, trainable=True):
        """
        Load itself pretrained LoRA adapter into the base model.
        """
        adapter_path = self._path_to_file() if adapter_path is not None else adapter_path
        self.base_peft_model.load_adapter(adapter_path, self.name, trainable=trainable)
        print(f"[ExpertAdapter] Loaded adapter from {adapter_path}")

    def unload_to_base_model(self):
        """
        Disable itself in the base model
        """
        curr_active_adapters = self.base_peft_model.active_adapters()
        if self.name in curr_active_adapters:
            curr_active_adapters = curr_active_adapters.pop(curr_active_adapters.index(self.name))
            print(f"[ExpertAdapter] Unloaded adapter {self.name}. Current active adapters: {curr_active_adapters}")
        else:
            print(f"[ExpertAdapter] Attempt to unload an inactive adapter {self.name}.")

    def save_adapter(self, ):
        """
        Save itself (only the LoRA adapter) weights to file.
        """
        save_path = self._path_to_file(path_only=True)
        os.makedirs(save_path, exist_ok=True)
        self.base_peft_model.save_pretrained(save_path, adapter_name=self.name)
        print(f"[ExpertAdapter] Saved adapter {self.name} to {save_path}")
