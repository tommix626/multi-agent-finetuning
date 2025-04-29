import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


# trainset is for part1 (clustering and fine-tuning)
# devset is for training of part2 (prompt-tuning, mixer)
# testset is for evaluation after part2 (overall evaluation)

class MMLUWrapper:
    def __init__(self, seed=42, expert_test_size=0.2, mixer_train_size=0.6, mixer_dev_size=0.2, mixer_test_size=0.2):
        self.seed = seed
        self.expert_test_size = expert_test_size
        self.mixer_train_size = mixer_train_size
        self.mixer_dev_size = mixer_dev_size
        self.mixer_test_size = mixer_test_size
        self.dataset = self._load_and_process()

    def _load_and_process(self):
        raw = load_dataset("cais/mmlu", "all")

        expert_split = raw["test"].train_test_split(test_size=self.expert_test_size, seed=self.seed)

        train_expert = self._map_choices_to_answer(expert_split["train"], "train_expert")
        dev_expert = self._map_choices_to_answer(expert_split["test"], "dev_expert")

        mixer_split = raw["auxiliary_train"].train_test_split(test_size=1.0 - self.mixer_train_size, seed=self.seed)
        mixer_dev_test_split = mixer_split["test"].train_test_split(test_size=self.mixer_test_size / (self.mixer_test_size + self.mixer_dev_size), seed=self.seed)

        train_mixer = self._map_choices_to_answer(mixer_split["train"], "dev")
        dev_mixer = self._map_choices_to_answer(mixer_dev_test_split["train"], "dev_mixer")
        test_mixer = self._map_choices_to_answer(mixer_dev_test_split["test"], "test_mixer")

        return DatasetDict({
            "train_expert": train_expert,
            "dev_expert": dev_expert,
            "train_mixer": train_mixer,
            "dev_mixer": dev_mixer,
            "test": test_mixer
        })

    def _map_choices_to_answer(self, dataset, split_name):
        def _mapper(example, idx):
            answer_index = example["answer"]
            answer_str = example["choices"][answer_index]
            return {
                "mapped_answer": answer_str,
                "uid": f"{split_name}-{idx:06d}"  # e.g., train-000001
            }
        return dataset.map(_mapper, with_indices=True)

    def get_dataset(self):
        return self.dataset

    def get_splits(self):
        return (self.dataset["train_expert"], self.dataset["dev_expert"],self.dataset["train_mixer"], self.dataset["dev_mixer"], self.dataset["test"])
    
    def get_train_expert(self):
        return self.dataset["train_expert"]

    def get_dev_expert(self):
        return self.dataset["dev_expert"]

    def get_train_mixer(self):
        return self.dataset["train_mixer"]

    def get_dev_mixer(self):
        return self.dataset["dev_mixer"]

    def get_test(self):
        return self.dataset["test"]
    
"""
Format:
{
    'input_ids': tokenized question (plus maybe answer later),
    'attention_mask': attention mask,
    'labels': masked labels (only answer tokens supervised),
    'uid': unique ID
}
"""
class mmluDataset(torch.utils.data.Dataset):

    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        This function is called by the DataLoader to get an instance of the data
        :param index:
        :return:
        """

        example = self.data[index]
        question = example["question"]
        #subject = example['subject']
        answer = example['mapped_answer']
        uid = example["uid"]


        # Tokenize the question (input)
        input_encoding = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True
        )

        # Tokenize the answer (label)
        answer_encoding = self.tokenizer.encode_plus(
            answer,
            add_special_tokens=False,
            max_length=10,  # Short max length for answers
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # === Fix: Align label with -100 pad at front ===
        labels = torch.full((self.max_len,), -100)  # Make labels same size as input_ids
        labels[:answer_encoding['input_ids'].shape[1]] = answer_encoding['input_ids'][0]

        # return Dict[str, torch.Tensor]
        return {
            'input_ids': input_encoding['input_ids'][0], 
            'attention_mask': input_encoding['attention_mask'][0],
            'labels': labels,
            'uid': uid
        }
    
def pre_process(model_name, batch_size, device, peft_config=None, mode='expert'):
    # download dataset
    print("Loading the dataset ...")
    mmlu_wrapper = MMLUWrapper()
    dataset = mmlu_wrapper.get_dataset()

    print("Loading the tokenizer...")
    mytokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    mytokenizer.pad_token = mytokenizer.eos_token  # for generation

    max_len = 512

    print(f"Loading the {mode} data into DS...")
    if mode == "expert":
        dataset_train = dataset['train_expert']
        dataset_dev = dataset['dev_expert']
    elif mode == "mixer":
        dataset_train = dataset['train_mixer']
        dataset_dev = dataset['dev_mixer']
    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'expert' or 'mixer'.")

    dataset_test = dataset['test']

    print(" >>>>>>>> Initializing the data loaders ... ")
    train_dataloader = DataLoader(
        mmluDataset(dataset_train, mytokenizer, max_len),
        batch_size=batch_size,
    )
    validation_dataloader = DataLoader(
        mmluDataset(dataset_dev, mytokenizer, max_len),
        batch_size=batch_size
    )
    test_dataloader = DataLoader(
        mmluDataset(dataset_test, mytokenizer, max_len),
        batch_size=batch_size
    )

    # Load the model
    print("Loading the model ...")
    pretrained_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # Apply PEFT (LoRA or other) if provided
    if peft_config is not None:
        pretrained_model = get_peft_model(pretrained_model, peft_config)

    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)

    return pretrained_model, train_dataloader, validation_dataloader, test_dataloader

 
def pre_process_data(model_name, batch_size, device, peft_config=None, mode='expert'):
    # download dataset
    print("Loading the dataset ...")
    mmlu_wrapper = MMLUWrapper()
    dataset = mmlu_wrapper.get_dataset()

<<<<<<< Updated upstream
    # print("Loding the data into DS...")
    # dataset_train = dataset['train']#.select(range(30))  # FIXME: temp fix for debug
    # print("training data",dataset_train)
    # dataset_dev = dataset['dev']
    # dataset_test = dataset['test']
=======
    print("Loding the data into DS...")
    dataset_train = dataset['train'].select(range(30))  # FIXME: temp fix for debug
    print("training data",dataset_train)
    dataset_dev = dataset['dev']
    dataset_test = dataset['test']
>>>>>>> Stashed changes

    print("Loading the tokenizer...")
    mytokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    mytokenizer.pad_token = mytokenizer.eos_token  # for generation

    max_len = 512

    print(f"Loding the {mode} data into DS...")
    if mode == "expert":
        dataset_train = dataset['train_expert']
        dataset_dev = dataset['dev_expert']
    elif mode == "mixer":
        dataset_train = dataset['train_mixer']
        dataset_dev = dataset['dev_mixer']
    else:
        raise ValueError(f"Unknown mode {mode}")

    dataset_test = dataset['test']

    print(" >>>>>>>> Initializing the data loaders ... ")
    train_dataloader = DataLoader(
        mmluDataset(dataset_train, mytokenizer, max_len),
        batch_size=batch_size,
    )
    validation_dataloader = DataLoader(
        mmluDataset(dataset_dev, mytokenizer, max_len),
        batch_size=batch_size
    )
    test_dataloader = DataLoader(
        mmluDataset(dataset_test, mytokenizer, max_len),
        batch_size=batch_size
    )
    return train_dataloader, validation_dataloader, test_dataloader
