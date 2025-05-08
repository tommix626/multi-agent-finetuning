import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, LoraConfig, TaskType, get_peft_model

from data.mmlu.mmludataset import pre_process
from evaluation._utils import evaluate_model

def load_model_and_tokenizer(base_model_name, checkpoint_path, peft_config):
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)
    model = get_peft_model(model, peft_config)
    
    # Load weights
    print(f"Loading model weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="EleutherAI/gpt-neo-125m")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--mode", type=str, default='full')
    args = parser.parse_args()

    # Setup LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,  # Note: True since this is evaluation
        r=args.rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj"]
    )

    # Load the model
    model = load_model_and_tokenizer(args.model, args.checkpoint_path, peft_config)
    model.to(args.device)

    # Load data
    _, _, val_loader, test_loader = pre_process(
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        peft_config=peft_config,
        mode=args.mode
    )

    total_metrics = {"loss": 0.0, "perplexity": 0.0, "accuracy": 0.0}
    total_batches = 0

    print("\nRunning Evaluation on Validation Set (per-batch)...")
    for batch in val_loader:
        batch_metrics = evaluate_model(model, batch, args.device)
        for k in total_metrics:
            total_metrics[k] += batch_metrics[k]
        total_batches += 1

    avg_metrics = {k: v / total_batches for k, v in total_metrics.items()}

    print("\n=== Validation Evaluation Summary ===")
    for k, v in avg_metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

if __name__ == "__main__":
    main()
