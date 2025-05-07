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
from sklearn.decomposition import PCA, TruncatedSVD
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

        # train_for_clustering_analysis = raw["test"].train_test_split(test_size=0.8, seed=self.seed)
        # train_for_clustering_analysis = train_for_clustering_analysis["train"]
        # train_for_clustering_analysis = self._map_choices_to_answer(train_for_clustering_analysis, "train_for_clustering_analysis")
        
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
            "train_13": train_13
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
                #"subject": subject
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
        Return a dictionary containing input_ids, attention_mask, and labels suitable for decoder-only models.
        The model should learn to generate the answer given the question.
        """
        example = self.data[index]
        question = example["question"]
        answer = example["mapped_answer"]
        uid = example["uid"]

        # Concatenate question and answer as a single string
        qa_pair = question.strip() + " " + answer.strip()

        # Tokenize the full input (question + answer)
        full_encoding = self.tokenizer(
            qa_pair,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = full_encoding["input_ids"][0]
        attention_mask = full_encoding["attention_mask"][0]

        # Tokenize only the question to find its token length
        question_encoding = self.tokenizer(
            question.strip(),
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        question_length = question_encoding["input_ids"].shape[1]

        # Construct labels: mask question part with -100, keep answer part
        labels = input_ids.clone()
        labels[:question_length] = -100  # Ignore question tokens in loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "uid": uid
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
    Cluster training examples based on concatenated question + answer.
    This version:
      • pulls texts/uids straight from train_loader.dataset.data
      • uses TruncatedSVD on the sparse TF–IDF
      • prints cluster sizes for quick sanity checks
    """

    print("Step 1: ")
    # 1) grab the raw data
    raw_data = getattr(train_loader.dataset, 'data', None)
    if raw_data is None:
        raise ValueError("Cannot find `dataset.data` on the provided DataLoader")
    
    texts = [f"{ex['question']} {ex['mapped_answer']}" for ex in raw_data]
    uids  = [ex['uid'] for ex in raw_data]

    if len(texts) == 0:
        raise ValueError("No examples to cluster!")

    assert len(texts) == len(uids), f"texts ({len(texts)}) vs uids ({len(uids)}) mismatch"

    print("Step 2: ")
    # 2) vectorize / embed
    if method == 'tfidf':
        vect = TfidfVectorizer(stop_words='english')
        X    = vect.fit_transform(texts)               # sparse
        n_comp = min(100, X.shape[1] - 1)
        if n_comp <= 0:
            raise ValueError("Too few TF–IDF features (<2) to run TruncatedSVD")
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        reduced = svd.fit_transform(X)                 # dense (n_samples × n_comp)
    elif method == 'sentence-transformer':
        model = SentenceTransformer('all-MiniLM-L6-v2')
        reduced = model.encode(texts, convert_to_numpy=True)
    else:
        raise ValueError(f"Unknown method `{method}`; choose 'tfidf' or 'sentence-transformer'")

    print("Step 3: ")
    # 3) cluster
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(reduced)
    assert len(labels) == len(uids), "Got a label count != number of examples"

    print("Step 4: ")
    # 4) group and debug-print
    clusters = [[] for _ in range(k)]
    for uid, lab in zip(uids, labels):
        clusters[lab].append(uid)

    return clusters


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

def get_clustered_dataloaders(model_name, batch_size, k=6, method='tfidf'):
    """
    Cluster train_12 data using KMeans and return a list of DataLoaders per cluster, plus dev/test DataLoaders.
    """

    mmlu_wrapper = MMLUWrapper()
    train_data = mmlu_wrapper.get_dataset()["train_12"]
    dev_data = mmlu_wrapper.get_dev()
    test_data = mmlu_wrapper.get_test()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    max_len = 512

    train_dataset = mmluDataset(train_data, tokenizer, max_len)
    full_loader = DataLoader(train_dataset, batch_size=batch_size)

    print("Performing KMeans clustering...")
    clustered_uids = k_means_clustering(full_loader, k=k, method=method)
    uid_to_index = {ex['uid']: i for i, ex in enumerate(train_data)}

    cluster_loaders = []
    for i, cluster_uids in enumerate(clustered_uids):
        subset = [train_data[uid_to_index[uid]] for uid in cluster_uids if uid in uid_to_index]
        cluster_dataset = mmluDataset(subset, tokenizer, max_len)
        cluster_loader = DataLoader(cluster_dataset, batch_size=batch_size)
        cluster_loaders.append(cluster_loader)

    dev_loader = DataLoader(mmluDataset(dev_data, tokenizer, max_len), batch_size=batch_size)
    test_loader = DataLoader(mmluDataset(test_data, tokenizer, max_len), batch_size=batch_size)

    return cluster_loaders, dev_loader, test_loader
