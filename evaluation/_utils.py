import torch

import os
import json
from typing import Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PeftConfig
from training.cluster_perplexity_trainer import TrainerConfig
from models.expert_cluster import ExpertCluster


def evaluate_model(model, batch, device):
    """
    Evaluate a PyTorch Model
    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader test_dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on
    :return accuracy
    """
    # load metrics
    total_loss = 0.0
    total_tokens = 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    total_correct = 0

    # turn model into evaluation mode
    model.eval()

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch["labels"].to(device)

    # forward pass
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
    print(f"model output.loss=", output.loss)
    logits = output.logits

    # Shift for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    num_tokens = (shift_labels != -100).sum()

    total_loss += loss.item()
    total_tokens += num_tokens.item()

    # Compute accuracy 
    predictions = torch.argmax(shift_logits, dim=-1)
    mask = shift_labels != -100
    correct = (predictions == shift_labels) & mask
    total_correct += correct.sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = total_correct / total_tokens


    print(f"metric loss=", avg_loss)
    # compute and return metrics
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy
    }


def load_expert_cluster_from_checkpoint(
    checkpoint_path: str,
    device: Optional[str] = None
) -> Tuple[ExpertCluster, TrainerConfig, AutoTokenizer]:
    """
    Load a fully configured ExpertCluster from a given checkpoint directory.

    Args:
        checkpoint_path (str): Path to checkpoint folder containing metadata.json and adapters.
        device (str, optional): If specified, overrides the device from training config.

    Returns:
        Tuple[ExpertCluster, TrainerConfig, AutoTokenizer]
    """
    metadata_path = os.path.join(checkpoint_path, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"[load] Missing metadata.json in {checkpoint_path}")

    # Load and reconstruct TrainerConfig
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    config_dict = metadata.get("training_config", {})
    config = TrainerConfig(**config_dict)
    if device:
        config.device = device

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(config.model_name)

    # Load LoRA PEFT config
    peft_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.exists(peft_config_path):
        raise FileNotFoundError(f"Missing adapter_config.json in {checkpoint_path}")
    peft_config = PeftConfig.from_pretrained(checkpoint_path)

    peft_model = get_peft_model(base_model, peft_config).to(config.device)

    # Instantiate and load expert cluster
    expert_cluster = ExpertCluster(
        base_model=peft_model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        **config.expert_cluster_kwargs()
    )
    expert_cluster.load_all_experts(checkpoint_path)

    return expert_cluster, config, tokenizer
