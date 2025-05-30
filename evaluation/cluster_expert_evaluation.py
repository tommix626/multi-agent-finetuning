"""
eval the model on eval dataset.

PYTHONPATH=. python evaluation/cluster_expert_evaluation.py --config=configs/new_save/new_save.yaml --epoch=7

"""
import os
import argparse
import re
from typing import Optional
import torch
from torch.utils.data import DataLoader
from evaluation._utils import evaluate_model, load_expert_cluster_from_checkpoint
from config import parse_config  # assume this loads and returns a Config object
from data.mmlu.mmludataset import pre_process_data
from models.expert_cluster import wrap_model_function_to_agent_function
import datetime
from pathlib import Path



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch to load (latest if None)")
    return parser.parse_args()

def main():
    args = parse_args()
    config = parse_config(args.config)
    trainer_id = config.trainer_id
    epoch_to_load = args.epoch
    config.device = "cpu" #FIXME: Temp override for testing

    # Paths
    checkpoints_root = config.save_dir
    checkpoint_folders = [
        d for d in os.listdir(checkpoints_root)
        if d.startswith(trainer_id)
    ]
    if not checkpoint_folders:
        raise ValueError(f"No checkpoint found for trainer ID: {trainer_id}")

    def extract_epoch(folder_name: str) -> Optional[int]:
        print(f"matching {folder_name}")
        match = re.search(r"-epoch-(\d+)", folder_name)
        print(f"matched {int(match.group(1)) if match else None}")
        return int(match.group(1)) if match else None

    # Filter only valid checkpoint folders with an epoch suffix
    checkpoint_folders = [
        d for d in os.listdir(checkpoints_root)
        if d.startswith(trainer_id)
        and os.path.isdir(os.path.join(checkpoints_root, d))
        and extract_epoch(d) is not None
    ]
    print(f"checkpoints={checkpoint_folders}")

    checkpoint_folders = sorted(checkpoint_folders, key=extract_epoch)
    if epoch_to_load is None:
        checkpoint_folder = checkpoint_folders[-1]
    else:
        checkpoint_folder = next((f for f in checkpoint_folders if f"-epoch-{epoch_to_load}" in f), None)
        if checkpoint_folder is None:
            raise ValueError(f"No checkpoint found for epoch {epoch_to_load}")

    checkpoint_path = os.path.join(checkpoints_root, checkpoint_folder)
    print(f"[Eval] Loading from checkpoint {checkpoint_path}")

    # Load cluster and tokenizer
    expert_cluster, _, tokenizer = load_expert_cluster_from_checkpoint(checkpoint_path, device=config.device)

    # Evaluation dataset
    train_classifier_loader, train_extra_loader, eval_loader, test_loader = pre_process_data(
        model_name=config.model_name,
        batch_size=config.batch_size,  # TODO: CONFIG change to 1 to be finegrained to documents.
        device=config.device if torch.cuda.is_available() else "cpu", #TODO: CONFIG: can force to be "cpu"
        peft_config=None,
        mode="expert"
    )

    # Evaluation loop
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"eval_metrics_{trainer_id}_epoch{extract_epoch(checkpoint_folder)}_{timestamp}.txt"
    

    num_exp = config.num_experts


    log_lines = [
        f"Model: {config.model_name}",
        f"Trainer ID: {trainer_id}",
        f"Checkpoint: {checkpoint_folder}",
        f"Epoch: {extract_epoch(checkpoint_folder)}",
        f"Evaluation Time: {timestamp}",
        f"Number of Experts: {num_exp}",
        "-" * 50
    ]

    # Collect individual expert metrics
    total_metrics = []
    for i in range(num_exp):
        total_metrics.append({"loss": 0.0, "perplexity": 0.0, "accuracy": 0.0})
    total_batches = 0

    for batch in train_classifier_loader:
        eval_fn = lambda m: evaluate_model(m, batch, config.device)
        wrapped_eval_function = wrap_model_function_to_agent_function(eval_fn) # expert -> res
        batch_metrics = expert_cluster.run_function_on_all_expert(wrapped_eval_function)  # call wrap_func(exp) to get res
        print(f"batch metrics: {batch_metrics}")

        for k in total_metrics[0]:
            for i in range(num_exp):
                total_metrics[i][k] += batch_metrics[i][k]

        total_batches += 1

    # Aggregate and save metrics
    with open(log_file, "w") as f:
        f.write("\n".join(log_lines) + "\n")

        for i in range(num_exp):
            avg_metrics = {k: v / total_batches for k, v in total_metrics[i].items()}

            print(f"\n=== Expert {i} Evaluation Metrics ===")
            f.write(f"\n=== Expert {i} ===\n")
            for k, v in avg_metrics.items():
                print(f"{k.capitalize()}: {v:.4f}")
                f.write(f"{k.capitalize()}: {v:.4f}\n")

if __name__ == "__main__":
    main()
