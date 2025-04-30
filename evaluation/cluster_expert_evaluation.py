import os
import argparse
from torch.utils.data import DataLoader
from evaluation._utils import evaluate_model, load_expert_cluster_from_checkpoint
from config import parse_config  # assume this loads and returns a Config object
from data.mmlu.mmludataset import pre_process_data
from models.expert_cluster import wrap_model_function_to_agent_function

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

    # Paths
    checkpoints_root = config.save_dir
    checkpoint_folders = [
        d for d in os.listdir(checkpoints_root)
        if d.startswith(trainer_id)
    ]
    if not checkpoint_folders:
        raise ValueError(f"No checkpoint found for trainer ID: {trainer_id}")

    checkpoint_folders = sorted(checkpoint_folders, key=lambda x: int(x.split("-epoch-")[-1]))
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
    _, _, eval_loader = pre_process_data(
        model_name=config.model_name,
        batch_size=config.batch_size,  # TODO: CONFIG change to 1 to be finegrained to documents.
        device=config.device,
        peft_config=None,
        mode="expert"
    )

    # Evaluation loop
    num_exp = config.num_experts
    total_metrics = []
    for i in range(num_exp):
        total_metrics.append({"loss": 0.0, "perplexity": 0.0, "accuracy": 0.0})
    total_batches = 0


    for batch in eval_loader:
        eval_function = lambda model: evaluate_model(model, batch, config.device)  # model -> res
        wrapped_eval_function = wrap_model_function_to_agent_function(eval_function) # expert -> res
        batch_metrics = expert_cluster.run_function_on_all_expert(wrapped_eval_function)  # call wrap_func(exp) to get res

        for k in total_metrics[0]:
            for i in range(num_exp):
                total_metrics[i][k] += batch_metrics[i][k]

        total_batches += 1

    for i in range(num_exp):
        avg_metrics = {k: v / total_batches for k, v in total_metrics[i].items()}

        print(f"\n=== Expert {i} Evaluation Metrics ===")
        for k, v in avg_metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")

if __name__ == "__main__":
    main()
