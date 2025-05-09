import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn as nn 
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import get_scheduler
from transformers import AutoModel, AutoModelForCausalLM
import argparse
import subprocess
import evaluate as evaluate
import matplotlib.pyplot as plt
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model
import time
import yaml

# Use a pipeline as a high-level helper
from transformers import pipeline

# pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

import os
# print(os.listdir("/home/cs601-lwang216/scr4-cs601-dkhasha1/cs601-lwang216/multi-agent-finetuning/data/mmlu"))

import sys
sys.path.insert(0, "/home/xwang397/scr4_jeisner1/tomwang/multi-agent-finetuning-lena/data/mmlu")
from mmludataset import pre_process, mmluDataset, get_clustered_dataloaders

"""""
The trainer class for training the clustering phase.
Goal is to call trainer.train() to perform training.

The Trainer manages:
- Config (learning rate, schedule, hyperparameters)
- Model and tokenizer instantiation
- Optimizer setup
- Expert cluster management
- Training logic: loss calculation and backpropagation
- Callback handling for modular events and logging
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os
import subprocess
from typing import Optional, List

import torch
from torch.utils.data import DataLoader
from peft import get_peft_model
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

@dataclass
class TrainerConfig:
    model_name: str = "EleutherAI/gpt-neo-125M"
    lr: float = 1e-4
    epochs: int = 3
    device: str = "cuda"
    trainer_id: str = "default-trainer"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Lora Base Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML or JSON config file.")
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif config_path.endswith(".json"):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError("Config file must be a .yaml, .yml, or .json file.")
    return config

def evaluate_model(model, dataloader, device):
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

    # iterate over the dataloader
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch["labels"].to(device)

        # forward pass
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels = labels)

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


    # compute and return metrics
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy
    }

def save_checkpoint(epoch: int, training_id, training_config, metrics, model, extra_folder_id=""):
    """Save training state after an epoch."""


    # create folder for this
    base_save_dir = f"checkpoints/kmeans-expert/epoch{epoch}/{training_id}{extra_folder_id}"
    os.makedirs(base_save_dir, exist_ok=True)
    
    model.save_pretrained(base_save_dir)

    # Save trainer-level metrics
    trainer_state = {
        "epoch": epoch,
        "metrics": metrics,
    }

    with open(os.path.join(base_save_dir, "trainer_state.json"), "w") as f:
        json.dump(trainer_state, f, indent=2)
    
    # NOTE: Save metadata
    def get_git_commit_hash() -> Optional[str]:
        """Return the current Git commit hash, or None if not inside a Git repository."""
        try:
            commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            return commit_hash
        except Exception:
            return None
    def to_jsonable(obj):
        """Try to convert an object to a JSON-serializable version."""
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        try:
            json.dumps(obj)  # check if json serializable
            return obj
        except (TypeError, OverflowError):
            return str(obj)  # fallback: store its string version
    config_dict = vars(training_config)
    jsonable_config = {k: to_jsonable(v) for k, v in config_dict.items()}

    metadata = {
        "trainer_id": trainer_id,
        "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_config": jsonable_config,
    }

    with open(os.path.join(base_save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Trainer] Saved checkpoint and metadata after epoch {epoch}.")

def train(mymodel, num_epochs, train_dataloader, validation_dataloader, test_dataloder, device, lr, model_name, rank, cluster, training_id, training_config):
    """ Train a PyTorch Module

    :param torch.nn.Module mymodel: the model to be trained
    :param int num_epochs: number of epochs to train for
    :param torch.utils.data.DataLoader train_dataloader: DataLoader containing training examples
    :param torch.utils.data.DataLoader validation_dataloader: DataLoader containing validation examples
    :param torch.device device: the device that we'll be training on
    :param float lr: learning rate
    :param string model_name: the name of the model
    :return None
    """

    # here, we use the AdamW optimizer. Use torch.optim.AdamW
    print(" >>>>>>>>  Initializing optimizer")
    
    optimizer = torch.optim.AdamW(mymodel.parameters(), lr=lr)
    # now, we set up the learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    loss = torch.nn.CrossEntropyLoss()
    
    epoch_list = []
    train_acc_list = []
    dev_acc_list = []

    for epoch in range(num_epochs):

        epoch_start_time = time.time()

        # put the model in training mode (important that this is done each epoch,
        # since we put the model into eval mode during validation)
        mymodel.train()

        print(f"Epoch {epoch + 1} training:")
        # Initialize counters
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        for index, batch in tqdm(enumerate(train_dataloader)):

            # get the input_ids, attention_mask, and labels from the batch and put them on the device
            # Hints: similar to the evaluate_model function
            print(f"Starting batch {index}...")

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # forward pass
            # name the output as `output`
            # Hints: refer to the evaluate_model function on how to get the predictions (logits)
            # - It's slightly different from the implementation in train of base_classification.py
            print(f"Batch {index} moved to device... shape: {input_ids.shape}")

            output = mymodel(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
            # compute the loss using the loss function
            # your code ends here
            loss = output.loss
            loss.backward()

            # update the model parameters depending on the model type
            print(f"Batch {index} loss.backward() done")

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            print(f"Batch {index} optimizer step done")

            logits = output.logits
            predictions = torch.argmax(logits, dim=-1)

            # Flatten prediction and padding tokens
            predictions = predictions.view(-1)
            labels = labels.view(-1)

            mask = labels != -100
            masked_preds = predictions[mask]
            masked_labels = labels[mask]

            batch_correct = (masked_preds == masked_labels).sum().item()
            batch_tokens = mask.sum().item()

            total_correct += batch_correct
            total_tokens += batch_tokens

            total_loss += loss.item() * batch_tokens
            print("iteration done...")


        # print evaluation metrics
        avg_train_loss = total_loss / total_tokens
        train_perplexity = torch.exp(torch.tensor(avg_train_loss)).item()
        avg_train_accuracy = total_correct / total_tokens

        print(f" ===> Epoch {epoch + 1}")
        print(f" - Average Training Loss: {avg_train_loss:.4f}")
        print(f" - Average Training Perplexity: {train_perplexity:.4f}")
        print(f" - Average Training Accuracy: {avg_train_accuracy:.4f}")
        train_acc_list.append(avg_train_accuracy)

        # normally, validation would be more useful when training for many epochs
        val_accuracy = evaluate_model(mymodel, validation_dataloader, device)
        print(f" - Average DEV metrics: loss={val_accuracy['loss']:.4f}, perplexity={val_accuracy['perplexity']:.4f}")
        dev_acc_list.append(val_accuracy['loss'])
        
        epoch_list.append(epoch)
        
        test_accuracy = evaluate_model(mymodel, test_dataloader, device)
        print(f" - Average test metrics: loss={test_accuracy['loss']:.4f}, perplexity={test_accuracy['perplexity']:.4f}")


        epoch_end_time = time.time()
        print(f"Epoch {epoch + 1} took {epoch_end_time - epoch_start_time} seconds")

        metrics = {
            "train_accuracy": avg_train_accuracy,
            "train_loss": avg_train_loss,
            "train_perplexity": train_perplexity,
            "dev_loss": val_accuracy["loss"],
            "dev_perplexity": val_accuracy["perplexity"],
            "test_loss": test_accuracy["loss"],
            "test_perplexity": test_accuracy["perplexity"],
        }

        save_checkpoint(epoch=epoch, training_id = training_id, training_config = training_config, metrics=metrics, model = mymodel)
        print(f"Model checkpoint saved")

# the entry point of the program
if __name__ == "__main__":

    args = parse_args()
    config_dict = load_config(args.config)

    # 1. Load hyperparameters
    model_name = config_dict.get("model_name", "EleutherAI/gpt-neo-125M")
    batch_size = config_dict.get("batch_size", 8)
    num_epochs = config_dict.get("num_epochs", 3)
    rank = config_dict.get("lora_r", 32)
    learning_rate = config_dict.get("learning_rate", 1e-4)
    device = config_dict.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    save_dir = config_dict.get("save_dir", "./checkpoints")
    default_trainer_id = f"default-{model_name}-batch-{batch_size}-r-{rank}-lr-{learning_rate}"
    trainer_id = config_dict.get("trainer_id", default_trainer_id)
    cluster = config_dict.get("cluster", "1")

    # 2. Define LoRA config
    print("initiating peft_config...")
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=config_dict.get("lora_r", 8), lora_alpha=config_dict.get("lora_alpha", 32), lora_dropout=config_dict.get("lora_dropout", 0.1), target_modules=["q_proj", "k_proj", "v_proj"])

    #load the data and models
    print("Loading model...")
    pretrained_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m", torch_dtype=torch.float16)
    pretrained_model = get_peft_model(pretrained_model, peft_config)

    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)

    print("Running pre_process...")
    cluster_loaders, validation_dataloader, test_dataloader = get_clustered_dataloaders(model_name="EleutherAI/gpt-neo-125m", batch_size=batch_size, k=cluster)  # Pass the LoRA configuration)
    
    model = 0
    for _,loader in enumerate(cluster_loaders):
        training_id=f"{trainer_id}-cluster#{model}"
        trainer_config = TrainerConfig(
            model_name=model_name,
            lr=learning_rate,
            epochs=num_epochs,
            device=device,
            trainer_id=training_id,
        )
        print(" >>>>>>>>  Starting training ... ")
        train(pretrained_model, num_epochs, loader, validation_dataloader, test_dataloader, device, learning_rate, model_name, rank, cluster, training_id, trainer_config)
        model = model + 1
