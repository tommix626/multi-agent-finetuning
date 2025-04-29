import torch
from torch.utils.data import DataLoader
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from peft import get_peft_model
import json
import os

from training.cluster_perplexity_trainer import TrainerConfig, ExpertTrainer

def main():
    # Load config
    trainer_id = "default-trainer"  # Change this if needed
    epoch_to_load = None  # Set to specific epoch or None to auto-load latest
    
    # Paths
    checkpoints_root = "checkpoints"
    checkpoint_folders = [
        d for d in os.listdir(checkpoints_root)
        if d.startswith(trainer_id)
    ]
    if not checkpoint_folders:
        raise ValueError(f"No checkpoint found for trainer ID: {trainer_id}")

    # Sort and pick checkpoint
    checkpoint_folders = sorted(checkpoint_folders, key=lambda x: int(x.split("-epoch-")[-1]))
    if epoch_to_load is None:
        checkpoint_folder = checkpoint_folders[-1]
    else:
        checkpoint_folder = next((f for f in checkpoint_folders if f"-epoch-{epoch_to_load}" in f), None)
        if checkpoint_folder is None:
            raise ValueError(f"No checkpoint found for epoch {epoch_to_load}")
    checkpoint_path = os.path.join(checkpoints_root, checkpoint_folder)
    print(f"[Eval] Loading from checkpoint {checkpoint_path}")

    # Load metadata
    with open(os.path.join(checkpoint_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    training_config = TrainerConfig(**metadata['training_config'])

    # Model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(training_config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(training_config.model_name)

    peft_config = load_peft_config(checkpoint_path)  # otherwise adjust

    model = get_peft_model(base_model, peft_config)
    model = model.to(training_config.device)

    # Load trained adapters
    from models.expert_cluster import ExpertCluster
    expert_cluster = ExpertCluster(
        base_model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        **training_config.expert_cluster_kwargs()
    )
    expert_cluster.load_all_experts(checkpoint_path)

    # Evaluation dataset
    from your_dataset_module import get_eval_dataset  # you need to define this
    eval_dataset = get_eval_dataset(tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False)

    # Evaluation loop
    model.eval()
    total_metrics = {"loss": 0.0, "perplexity": 0.0, "accuracy": 0.0}
    total_batches = 0

    for batch in eval_loader:
        # Select expert
        expert = expert_cluster.delegate_to_expert(batch)
        batch_metrics = evaluate_model(expert.adapter.model, batch, training_config.device)

        for k in total_metrics.keys():
            total_metrics[k] += batch_metrics[k]
        total_batches += 1

    # Average the metrics
    avg_metrics = {k: v / total_batches for k, v in total_metrics.items()}

    print("\n=== Evaluation Metrics ===")
    for k, v in avg_metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

if __name__ == "__main__":
    main()
