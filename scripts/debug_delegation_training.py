import os
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from peft import LoraConfig, PeftModel, get_peft_model, peft_model
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from transformers.integrations.peft import PeftAdapterMixin

# 1. Set up base model & tokenizer
model_name = "bert-base-uncased"
tokenizer   = AutoTokenizer.from_pretrained(model_name)
ori_model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 2. Define a LoRA config
lora_config = LoraConfig(
    task_type="SEQ_CLS",      # classification
    inference_mode=False,     # training
    r=8,                      # LoRA rank
    lora_alpha=16,            # scaling factor
    lora_dropout=0.05,        # dropout
    target_modules=["query", "key"],  # inject into these submodules
)

model = PeftModel(ori_model, peft_config=lora_config)
# 3. Create k more adapters
k = 3
adapter_names = [f"lora_adapter_{i}" for i in range(k)]
for name in adapter_names:
    model.add_adapter(adapter_name=name, peft_config=lora_config)    # add fresh adapter  [oai_citation_attribution:0‡Hugging Face](https://huggingface.co/docs/transformers/main/en/main_classes/peft)
# By default, add_adapter also does model.set_adapter(name) for you  [oai_citation_attribution:1‡Hugging Face](https://huggingface.co/docs/transformers/main/en/main_classes/peft)
print(model.active_adapters)
# 3) Now *enable* all adapters and *train* each of them so their weights require gradients
# model.enable_adapters()                                   # make all adapters “active”  [oai_citation_attribution:0‡Hugging Face](https://huggingface.co/docs/transformers/main/en/peft)
print(model.active_adapters)
# 4. Freeze base model, train only adapter params ( NOTE: by default in peft model, no need)
# for param in model.base_model.parameters():
#     param.requires_grad = False
# Gather only adapter parameters
for n,p in model.named_parameters():
    print(f"parameter {n}, require grad: {p.requires_grad}")

print(model.print_trainable_parameters())  # should list sum over all K adapters only


# 5) Build a single optimizer over *all* adapter params
optimizer = AdamW(
    [p for n,p in model.named_parameters() if "lora" in n],
    lr=1e-4,
)

# 5. Dummy DataLoader (replace with your real one)
#    Each batch is a dict: {"input_ids": ..., "attention_mask": ..., "labels": ...}
dataloader = DataLoader(
    [{"input_ids": tokenizer("Hello world", return_tensors="pt")["input_ids"].squeeze(),
      "attention_mask": tokenizer("Hello world", return_tensors="pt")["attention_mask"].squeeze(),
      "labels": torch.tensor(1)}] * 1000,
    batch_size=8,
)

# 6. Training loop: randomly pick an adapter each step
num_epochs = 2
num_training_steps = num_epochs * len(dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer,
    num_warmup_steps=0, num_training_steps=num_training_steps
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()

for epoch in range(num_epochs):
    for batch in dataloader:
        # 6a. Randomly choose one adapter
        adapter = random.sample(adapter_names,1,counts=[1,5,10])[0]
        model.set_adapter(adapter)                        # switch active adapter  [oai_citation_attribution:2‡Hugging Face](https://huggingface.co/docs/transformers/main/en/main_classes/peft)
        print("active:",model.active_adapter)

        # 6b. Forward / backward
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        # print the loss for this adapter
        print(f"[Epoch {epoch+1}] Adapter {adapter} → loss: {loss.item():.4f}")
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1} done")

# 7. Helpers to save / load individual adapters:

def save_adapter(model, adapter_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.set_adapter(adapter_name)                      # activate the adapter to save  [oai_citation_attribution:3‡Hugging Face](https://huggingface.co/docs/transformers/main/en/main_classes/peft)
    model.save_pretrained(save_dir)                      # saves adapter_config.json + adapter_model.*  [oai_citation_attribution:4‡Hugging Face](https://huggingface.co/docs/transformers/main/en/peft?utm_source=chatgpt.com)

def load_adapter(model, adapter_name, load_dir):
    # loadafter reloading base model:
    model.load_adapter(
        peft_model_id=load_dir,
        adapter_name=adapter_name,
        is_trainable=True,    # make sure it’s trainable again if you’ll continue training  [oai_citation_attribution:5‡Hugging Face](https://huggingface.co/docs/transformers/main/en/main_classes/peft)
    )

# Example usage:
#   save_adapter(model, "lora_adapter_0", "./checkpoints/adapter_0")
#   ...
#   base = AutoModelForSequenceClassification.from_pretrained(model_name)
#   load_adapter(base, "lora_adapter_0", "./checkpoints/adapter_0")
