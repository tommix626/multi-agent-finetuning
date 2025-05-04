import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model
import json
import numpy as np
import pandas as pd
import requests
import argparse
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# trainset is for part1 (clustering and fine-tuning)
# devset is for training of part2 (prompt-tuning, mixer)
# testset is for evaluation after part2 (overall evaluation)

class MMLUWrapper:
    def __init__(self, seed=42, expert_train_size=0.5, mixer_train_size=0.3, dev_size=0.1, test_size=0.1):
        self.seed = seed
        self.expert_train_size = expert_train_size
        self.mixer_train_size = mixer_train_size
        self.dev_size = dev_size
        self.test_size = test_size
        self.dataset = self._load_and_process()

    def _load_and_process(self):
        raw = load_dataset("cais/mmlu", "all")

        # expert_split = raw["test"].train_test_split(test_size=self.expert_test_size, seed=self.seed)
        # train_expert = self._map_choices_to_answer(expert_split["train"], "train_expert")
        # dev_expert = self._map_choices_to_answer(expert_split["test"], "dev_expert")

        train_for_clustering_analysis = raw["test"].train_test_split(test_size=0.8, seed=self.seed)
        train_for_clustering_analysis = train_for_clustering_analysis["train"]
        train_for_clustering_analysis = self._map_choices_to_answer(train_for_clustering_analysis, "train_for_clustering_analysis")
        
        train_classifier = self._map_choices_to_answer(raw["test"], "train_classifier")

        aux_split = raw["auxiliary_train"].train_test_split(test_size=1.0 - self.expert_train_size, seed=self.seed)
        train_expert = self._map_choices_to_answer(aux_split["train"], "train_expert")

        mixer_split = aux_split["test"].train_test_split(test_size= (self.dev_size + self.test_size) / (self.dev_size + self.test_size + self.mixer_train_size), seed=self.seed)

        train_mixer = self._map_choices_to_answer(mixer_split["train"], "train_mixer")
        mixer_dev_test_split = mixer_split["test"].train_test_split(test_size=self.test_size / (self.dev_size + self.test_size), seed=self.seed)

        dev = self._map_choices_to_answer(mixer_dev_test_split["train"], "dev")
        test = self._map_choices_to_answer(mixer_dev_test_split["test"], "test")

        train_full = concatenate_datasets([train_classifier, train_expert, train_mixer])
        train_13 = concatenate_datasets([train_classifier, train_mixer])
        train_12 = concatenate_datasets([train_classifier, train_expert])

        return DatasetDict({
            "train_classifier": train_classifier,
            "train_expert": train_expert,
            "train_mixer": train_mixer,
            "dev": dev,
            "test": test,
            "train_full": train_full,
            "train_12": train_12,
            "train_13": train_13,
            "train_for_clustering_analysis": train_for_clustering_analysis
        })

    def _map_choices_to_answer(self, dataset, split_name):
        def _mapper(example, idx):
            answer_index = example["answer"]
            answer_str = example["choices"][answer_index]
            subject = example["subject"]
            question = example['question']
            return {
                "question": question,
                "mapped_answer": answer_str,
                "uid": f"{split_name}-{idx:06d}",  # e.g., train-000001
                "subject": subject
            }
        return dataset.map(_mapper, with_indices=True)
    
    def get_analysis_dataset(self):
        return self.dataset["train_for_clustering_analysis"]

    def get_dataset(self):
        return self.dataset

    def get_splits(self):
        return (
            self.dataset["train_expert"],
            self.dataset["train_mixer"],
            self.dataset["train_classifier"],
            self.dataset["train_12"],
            self.dataset["train_13"],
            self.dataset["dev"],
            self.dataset["test"],
        )
    def get_train_expert(self):
        return self.dataset["train_expert"]

    def get_train_mixer(self):
        return self.dataset["train_mixer"]

    def get_train_classifier(self):
        return self.dataset["train_classifier"]

    def get_dev(self):
        return self.dataset["dev"]

    def get_test(self):
        return self.dataset["test"]
    
    def get_train_full(self):
        return self.dataset["train_full"]
    
    def get_subject_subset(self, split_name: str, subject: str):
        """
        return the data of a subject from a split.
        """
        if split_name not in self.dataset:
            raise ValueError(f"Invalid split name: {split_name}")
        return self.dataset[split_name].filter(lambda x: x["subject"] == subject)

    # def get_by_uid(self, uid: str):
    #     """
    #     Return the example with the given UID from the entire dataset.
    #     """
    #     for split in self.dataset:
    #         result = self.dataset[split].filter(lambda x: x["uid"] == uid)
    #         if len(result) > 0:
    #             return result[0]
    #     raise ValueError(f"UID {uid} not found in any dataset split.")
    
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
        subject = example['subject']
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
            max_length=self.max_len,
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
    
def _tfidf_cluster(texts, k):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    pca = PCA(n_components=min(100, tfidf_matrix.shape[1] - 1))
    reduced = pca.fit_transform(tfidf_matrix.toarray())
    kmeans = KMeans(n_clusters=k, random_state=42)
    return kmeans.fit_predict(reduced)


def _sentence_transformer_cluster(texts, k):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    kmeans = KMeans(n_clusters=k, random_state=42)
    return kmeans.fit_predict(embeddings)


def k_means_clustering(train_loader, k=5, method='tfidf'):
    """
    Cluster training examples based on concatenated question and answer strings.

    Args:
        train_loader: DataLoader for training data (no shuffle ideally).
        k: Number of clusters.
        method: 'tfidf' or 'sentence-transformer'.

    Returns:
        List of k clusters, each a list of UIDs.
    """
    # Build a mapping from uid to concatenated text
    dataset = train_loader.dataset.data  
    uid2text = {ex['uid']: f"{ex['question']} {ex['mapped_answer']}" for ex in dataset}

    # Extract texts in the order of the loader
    texts, uids = [], []
    for batch in train_loader:
        for uid in batch['uid']:
            texts.append(uid2text[uid])
            uids.append(uid)

    # Perform clustering
    if method == 'tfidf':
        labels = _tfidf_cluster(texts, k)
    elif method == 'sentence-transformer':
        labels = _sentence_transformer_cluster(texts, k)
    else:
        raise ValueError("Unsupported clustering method. Choose 'tfidf' or 'sentence-transformer'.")

    # Group UIDs by cluster
    clustered_uids = [[] for _ in range(k)]
    for uid, label in zip(uids, labels):
        clustered_uids[label].append(uid)
    return clustered_uids


def pre_process(model_name, batch_size, device, peft_config=None, mode='expert'):
    # download dataset
    print("Loading the dataset ...")
    mmlu_wrapper = MMLUWrapper()
    dataset = mmlu_wrapper.get_dataset()

    print("Loading the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token  # for generation

    max_len = 512

    print(f"Loading the {mode} data into DS...")
    if mode == "expert":
        train_classifier_data  = dataset["train_classifier"]
        train_expert_data  = dataset["train_expert"]
        train_classifier_loader = DataLoader(
            mmluDataset(train_classifier_data, tokenizer, max_len),
            batch_size=batch_size,
        )
        train_expert_loader = DataLoader(
            mmluDataset(train_expert_data, tokenizer, max_len),
            batch_size=batch_size,
        )
    elif mode == "mixer":
        train_data  = dataset["train_mixer"]
        train_loader = DataLoader(
            mmluDataset(train_data, tokenizer, max_len),
            batch_size=batch_size,
        )
    elif mode == "full":
        train_data  = dataset["train_full"]
        train_loader = DataLoader(
            mmluDataset(train_data, tokenizer, max_len),
            batch_size=batch_size,
        )
    elif mode == "12":
        train_data  = dataset["train_12"]
        train_loader = DataLoader(
            mmluDataset(train_data, tokenizer, max_len),
            batch_size=batch_size,
        )
    elif mode == "13":
        train_data  = dataset["train_13"]
        train_loader = DataLoader(
            mmluDataset(train_data, tokenizer, max_len),
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'expert', 'mixer', or 'full'.")
    
    dev_loader = DataLoader(
        mmluDataset(dataset["dev"], tokenizer, max_len),
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        mmluDataset(dataset["test"], tokenizer, max_len),
        batch_size=batch_size,
    )

    # Load the model
    print("Loading the model ...")
    pretrained_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # Apply PEFT (LoRA or other) if provided
    if peft_config is not None:
        pretrained_model = get_peft_model(pretrained_model, peft_config)

    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)

    if mode == "expert":
        return pretrained_model, train_classifier_loader, train_expert_loader, dev_loader, test_loader
    else:
        return pretrained_model, train_loader, dev_loader, test_loader

 
def pre_process_data(model_name, batch_size, device, peft_config=None, mode='expert'):
    print("Loading the dataset ...")
    mmlu_wrapper = MMLUWrapper()
    dataset = mmlu_wrapper.get_dataset()

    print("Loading the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token  # for generation

    max_len = 512

    print(f"Loading the {mode} data into Dataset objects...")
    if mode == "expert":
        train_classifier_data = dataset['train_classifier']
        train_expert_data = dataset['train_expert']
        train_classifier_loader = DataLoader(
            mmluDataset(train_classifier_data, tokenizer, max_len),
            batch_size=batch_size
        )
        train_expert_loader = DataLoader(
            mmluDataset(train_expert_data, tokenizer, max_len),
            batch_size=batch_size
        )
    elif mode == "mixer":
        train_data = dataset['train_mixer']
        train_loader = DataLoader(
            mmluDataset(train_data, tokenizer, max_len),
            batch_size=batch_size
        )
    elif mode == "full":
        train_data = dataset['train_full']
        train_loader = DataLoader(
            mmluDataset(train_data, tokenizer, max_len),
            batch_size=batch_size
        )
    elif mode == "12":
        train_data  = dataset["train_12"]
        train_loader = DataLoader(
            mmluDataset(train_data, tokenizer, max_len),
            batch_size=batch_size,
        )
    elif mode == "13":
        train_data  = dataset["train_13"]
        train_loader = DataLoader(
            mmluDataset(train_data, tokenizer, max_len),
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'expert', 'mixer', or 'full'.")

    dev_data = dataset['dev']
    test_data = dataset['test']

    dev_loader = DataLoader(
        mmluDataset(dev_data, tokenizer, max_len),
        batch_size=batch_size
    )
    test_loader = DataLoader(
        mmluDataset(test_data, tokenizer, max_len),
        batch_size=batch_size
    )

    if mode == "expert":
        return train_classifier_loader, train_expert_loader, dev_loader, test_loader
    else:
        return train_loader, dev_loader, test_loader


def pre_process_subject_data(model_name, batch_size, split_name='train_expert', subject='abstract_algebra'):
    print("Loading the dataset ...")
    mmlu_wrapper = MMLUWrapper()
    subject_dataset = mmlu_wrapper.get_subject_subset(split_name=split_name, subject=subject)

    print("Loading the tokenizer...")
    mytokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    mytokenizer.pad_token = mytokenizer.eos_token  # for generation

    max_len = 512

    print(" >>>>>>>> Initializing the data loaders ... ")
    subject_dataloader = DataLoader(
        mmluDataset(subject_dataset, mytokenizer, max_len),
        batch_size=batch_size,
    )
    return subject_dataloader

