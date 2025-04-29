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
sys.path.insert(0, "/home/cs601-lwang216/scr4-cs601-dkhasha1/cs601-lwang216/multi-agent-finetuning/data/mmlu")
from mmludataset import pre_process, mmluDataset

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
        total_correct += correct.sum

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = total_correct / total_tokens


    # compute and return metrics
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy
    }


def train(mymodel, num_epochs, train_dataloader, validation_dataloader, test_dataloder, device, lr, model_name, rank):
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
            # update metrics
            # train_accuracy.add_batch(predictions=predictions, references=labels)

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

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name.replace('/', '_')}_r={rank}_final.pt")

    torch.save({
        'epoch': num_epochs,
        'model_state_dict': mymodel.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'final_train_loss': avg_train_loss,
        'final_train_perplexity': train_perplexity,
        'final_train_accuracy': avg_train_accuracy,
        'final_val_loss': val_accuracy['loss'],
        'final_val_perplexity': val_accuracy['perplexity'],
        'final_test_loss': test_accuracy['loss'],
        'final_test_perplexity': test_accuracy['perplexity'],
    }, checkpoint_path)

    print(f"Final model checkpoint saved at {checkpoint_path}")

# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_subset", action='store_true')
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="EleutherAI/gpt-neo-125m")
    parser.add_argument("--rank", type=str, default=8)
    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    assert type(args.small_subset) == bool, "small_subset must be a boolean"
    #load the data and models
    print("Loading model...")
    pretrained_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m", torch_dtype=torch.float16)
    print("initiating peft_config...")
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.rank, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj"])
    print("Running pre_process...")
    pretrained_model, train_dataloader, validation_dataloader, test_dataloader = pre_process(model_name="EleutherAI/gpt-neo-125m", batch_size=args.batch_size, device="cuda", peft_config=peft_config, mode = 'full')  # Pass the LoRA configuration)

    print(" >>>>>>>>  Starting training ... ")
    train(pretrained_model, args.num_epochs, train_dataloader, validation_dataloader, test_dataloader, args.device, args.lr, args.model, args.rank)

    val_accuracy = evaluate_model(pretrained_model, validation_dataloader, args.device)
    print(f" - Average DEV metrics: accuracy={val_accuracy['accuracy']}, perplexity={val_accuracy['perplexity']}, loss={val_accuracy['loss']}")

    test_accuracy = evaluate_model(pretrained_model, test_dataloader, args.device)
    print(f" - Average TEST metrics: accuracy={test_accuracy['accuracy']}, perplexity={test_accuracy['perplexity']}, loss={test_accuracy['loss']}")
