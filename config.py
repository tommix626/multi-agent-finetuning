import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass
class Config:
    # Core model and training settings
    model_name: str
    batch_size: int
    num_epochs: int
    num_experts: int
    learning_rate: float
    device: str = "cuda"

    # LoRA / PEFT specific settings
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Misc training control
    save_dir: str = "./checkpoints"
    trainer_id: str = "default-trainer"
    selection_temperature: float = 1.0
    early_stopping_patience: int = 2

    # Metadata for tracking
    config_path: Optional[str] = None  # populated by loader

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Optional[str] = None):
        path = path or self.config_path
        if path is None:
            raise ValueError("Cannot save config: no path provided or set in object.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f)

    def summary(self):
        print("\n[Config Summary]")
        for k, v in self.to_dict().items():
            print(f"  {k}: {v}")
        print()

    def with_override(self, **kwargs) -> "Config":
        """Return a new Config object with given field overrides."""
        new_data = self.to_dict()
        new_data.update(kwargs)
        return Config(**new_data)


# ------------- Loader -------------
def parse_config(path: str) -> Config:
    """
    Load a Config object from YAML.
    Automatically fills missing fields with defaults.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw_data = yaml.safe_load(f)

    default_fields = {f.name for f in field(Config)}
    unknown_keys = set(raw_data.keys()) - default_fields
    if unknown_keys:
        print(f"[Warning] Unknown config fields in {path}: {unknown_keys}")

    return Config(**raw_data, config_path=path)


# ------------- CLI Integration (Optional) -------------
def config_from_cli(path: str, overrides: Dict[str, Any] = {}) -> Config:
    base = parse_config(path)
    return base.with_override(**overrides)

