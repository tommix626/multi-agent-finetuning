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
    
class mmluDataset(torch.utils.data.Dataset):
    """
    Dataset for the dataset of BoolQ questions and answers
    """

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
        subject = example['subject']
        answer = example['mapped_answer']


        # input encoding for your model
        input_encoding = question

        # encode_plus will encode the input and return a dictionary of tensors
        encoded_review = self.tokenizer.encode_plus(
            input_encoding,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        
        return {
            'input_ids': encoded_review['input_ids'][0], 
            'attention_mask': encoded_review['attention_mask'][0],
            'answer': torch.tensor(answer, dtype=torch.long)  
        }
    
def pre_process(model_name, batch_size, device, peft_config=None):
    # download dataset
    print("Loading the dataset ...")
    mmlu_wrapper = MMLUWrapper()
    dataset = mmlu_wrapper.get_dataset()

    print("Loding the data into DS...")
    dataset_train = dataset['train']
    dataset_dev = dataset['dev']
    dataset_test = dataset['test']

    print("Loading the tokenizer...")
    mytokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    mytokenizer.pad_token = mytokenizer.eos_token  # for generation

    max_len = 512

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
