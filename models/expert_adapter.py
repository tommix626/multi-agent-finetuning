"""""
The LoRA Adapter class for base model finetuning.
Encapsulates loading, unloading, saving of the adapter logic cleanly.
"""

import os
from typing import Optional, Union, List

import torch
from peft import (
    LoraConfig,
    PeftConfig,
    PeftMixedModel,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer


class ExpertAdapter:
    def __init__(
        self,
        base_peft_model: Union[PeftModel, PeftMixedModel],
        config: PeftConfig,
        name: str,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        target_modules: Optional[List[str]] = None,
        force_new: bool = False,
        verbose: bool = True,
    ):
        self.device = device
        self.base_peft_model = base_peft_model
        self.tokenizer = tokenizer
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.name = name
        self.peft_config = config
        self.verbose = verbose

        self._init_new_or_load(force_new)

    def _init_new_or_load(self, force_new: bool):
        """Either attach a fresh adapter or load from checkpoint."""
        if force_new:
            self.attach_raw()
        else:
            self.load()

    def get_save_dir(self) -> str:
        """Return the directory to save/load this adapter."""
        return os.path.join("checkpoints", self.name)

    def get_checkpoint_save_dir(self) -> str:
        """Return the directory to save/load this adapter."""
        return self.name



    def attach_raw(self):
        """Attach a new, randomly initialized LoRA adapter to the base model."""
        self.base_peft_model.add_adapter(self.name, self.peft_config)
        if self.verbose:
            print(f"[ExpertAdapter] Adding new adapter '{self.name}' to base model.")

    def load(self, adapter_path: Optional[str] = None, trainable: bool = True):
        """Load the pretrained adapter weights into the base model."""
        adapter_path = adapter_path or self.get_save_dir()

        if not os.path.isdir(adapter_path):
            if self.verbose:
                print(f"[ExpertAdapter] No adapter found at {adapter_path}. Initializing new adapter...")
            self.attach_raw()
            return

        self.base_peft_model.load_adapter(adapter_path, self.name, trainable=trainable)
        if self.verbose:
            print(f"[ExpertAdapter] Loaded adapter '{self.name}' from {adapter_path}.")

    def unload(self):
        """Disable the currently active adapter."""
        if self.name in self.base_peft_model.active_adapters:
            self.base_peft_model.disable_adapter()
            if self.verbose:
                print(f"[ExpertAdapter] Disabled adapter '{self.name}'.")
        else:
            if self.verbose:
                print(f"[ExpertAdapter] Adapter '{self.name}' was not active. No unload necessary.")

    def save_adapter(self, agent_save_dir: Optional[str]):
        """Save the adapter weights to file."""
        save_path = self.get_save_dir() if agent_save_dir is None else agent_save_dir
        os.makedirs(save_path, exist_ok=True)

        # self.base_peft_model.save_pretrained(save_path) # FIXME:
        if self.verbose:
            print(f"[ExpertAdapter] Saved adapter '{self.name}' to {save_path}.")

    def activate(self):
        """Activate this adapter inside the PEFT model for training/inference."""
        self.base_peft_model.set_adapter(self.name)
        if self.verbose:
            print(f"[ExpertAdapter] Activated adapter '{self.name}'.")
