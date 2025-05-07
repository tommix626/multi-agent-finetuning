import torch

import os
import json
from typing import Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PeftConfig
from training.cluster_perplexity_trainer import TrainerConfig
from models.expert_cluster import ExpertCluster


def evaluate_model(model, batch, device, mode=1):
    """
    Evaluate a PyTorch Model
    :param torch.nn.Module model: the model to be evaluated
    :param batch: one batch from DataLoader
    :param torch.device device: device to run on
    :param int mode: 0 for token‐accuracy (default), 1 for MCQ‐accuracy
    :return dict(loss, perplexity, accuracy)
    """
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')

    model.eval()

    # forward pass
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch["labels"].to(device)
    print(f"input shape={input_ids.shape}")
    print(f"attn mask shape={attention_mask.shape}")
    print(f"label shape={labels.shape}")
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    logits = output.logits

    # Shift for next‐token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    num_tokens = (shift_labels != -100).sum()

    total_loss += loss.item()
    total_tokens += num_tokens.item()

    # token‐level correct
    preds = torch.argmax(shift_logits, dim=-1)
    mask = shift_labels != -100
    total_correct += ((preds == shift_labels) & mask).sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    # --- NEW BRANCH ---
    if mode == 1:
        # MCQ‐style accuracy
        accuracy = mcq_acc(model, batch, device)
    else:
        # token‐level accuracy
        accuracy = total_correct / total_tokens

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy
    }

def mcq_acc(model, batch, device):
    """
    For each example in the batch, compute the loss for each choice
    and count a correct prediction if the lowest‐loss choice matches the true labels.
    Returns: batch_accuracy (float in [0,1])
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    true_labels = batch["labels"].to(device)                # shape (B, L)
    all_choice_labels = batch["all_choice_labels"]          # list of lists: len=B, each inner list len=C

    batch_size = input_ids.size(0)
    correct = 0

    model.eval()
    with torch.no_grad():
        for i in range(batch_size):
            losses = []
            for choice_labels in all_choice_labels[i]:
                cl = choice_labels.to(device).unsqueeze(0)   # (1, L)
                out = model(
                    input_ids = input_ids[i].unsqueeze(0),
                    attention_mask = attention_mask[i].unsqueeze(0),
                    labels = cl
                )
                # out.loss is avg over non-ignored tokens
                losses.append(out.loss.item())

            # pick the choice with lowest loss
            pred_idx = int(np.argmin(losses))
            # find which index in all_choice_labels matches the true_labels
            true_idx = next(
                j for j, cl in enumerate(all_choice_labels[i])
                if torch.equal(cl.to(device), true_labels[i])
            )
            if pred_idx == true_idx:
                correct += 1

    return correct / batch_size

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
