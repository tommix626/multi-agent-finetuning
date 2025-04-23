import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# trainset is for part1 (clustering and fine-tuning)
# devset is for training of part2 (prompt-tuning, mixer)
# testset is for evaluation after part2 (overall evaluation)

class MMLUWrapper:
    def __init__(self, seed=42, test_size=0.2):
        self.seed = seed
        self.test_size = test_size
        self.tokenizer = None
        self.max_len = 512
        self.dataset = self._load_and_process()

    def _load_and_process(self):
        raw = load_dataset("cais/mmlu", "all")

        train_set  = self._map_choices_to_answer(raw["test"])
        aux_data = self._map_choices_to_answer(raw["auxiliary_train"])

        aux_split = aux_data.train_test_split(
            test_size=self.test_size, seed=self.seed
        )
        dev_set = aux_split["train"]
        test_set = aux_split["test"]

        return DatasetDict({
            "train": train_set,
            "dev": dev_set,
            "test": test_set
        })

    def _map_choices_to_answer(self, dataset):
        def _mapper(example):
            answer_index = example["answer"]
            answer_str = example["choices"][answer_index]
            return {"mapped_answer": answer_str}
        return dataset.map(_mapper)
    
    def set_tokenizer(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_dataset(self):
        assert self.tokenizer is not None, "Call `set_tokenizer()` first."

        def _tokenize(example):
            input_text = example["question"].strip()
            target_text = example["mapped_answer"].strip()

            input_enc = self.tokenizer(
                input_text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True
            )
            label_enc = self.tokenizer(
                target_text,
                max_length=32,
                padding="max_length",
                truncation=True
            )

            return {
                "input_ids": input_enc["input_ids"],
                "attention_mask": input_enc["attention_mask"],
                "labels": label_enc["input_ids"]
            }

        self.dataset = self.dataset.map(_tokenize, remove_columns=self.dataset["train"].column_names)

    def get_dataset(self):
        return self.dataset

    def get_splits(self):
        return self.dataset["train"], self.dataset["dev"], self.dataset["test"]
    
    def get_train(self):
        return self.dataset["train"]

    def get_dev(self):
        return self.dataset["dev"]

    def get_test(self):
        return self.dataset["test"]
    
    def prepare_dataloaders(self, batch_size=32):
        assert self.tokenizer is not None, "Call `set_tokenizer()` before preparing dataloaders."

        def build_dataloader(split):
            return DataLoader(
                mmluDataset(self.dataset[split]),
                batch_size=batch_size,
                shuffle=(split == "train")
            )

        return {
            "train": build_dataloader("train"),
            "dev": build_dataloader("dev"),
            "test": build_dataloader("test"),
        }

class mmluDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long)
        }

def pre_process(model_name, batch_size, device, type='auto'):
    print("Loading the dataset ...")
    mmlu_wrapper = MMLUWrapper()
    
    print("Loading the tokenizer ...")
    mmlu_wrapper.set_tokenizer(model_name)
    tokenizer = mmlu_wrapper.tokenizer

    print("Initializing the data loaders ...")
    dataloaders = mmlu_wrapper.prepare_dataloaders(batch_size=batch_size)
    train_dataloader = dataloaders["train"]
    validation_dataloader = dataloaders["dev"]
    test_dataloader = dataloaders["test"]

    print("Loading the model ...")
    pretrained_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set task type and PEFT config (LoRA)
    peft_config = LoraConfig(
        task_type=TaskType.QUESTION_ANS,  # Adjust if needed
        inference_mode=False,
        r=6,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj"]  # Adjust to your model
    )
    pretrained_model = get_peft_model(pretrained_model, peft_config)

    print(f"Moving model to device: {device}")
    pretrained_model.to(device)

    return pretrained_model, train_dataloader, validation_dataloader, test_dataloader
