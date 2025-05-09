import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

import json
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from data.mmlu.mmludataset import MMLUWrapper
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
import numpy as np

def load_cluster_assignments(json_path):
    with open(json_path, 'r') as f:
        cluster_assignments = json.load(f)
    return cluster_assignments

def build_uid_maps(mmlu_wrapper, expert_ids, base_path):
    train_expert = mmlu_wrapper.get_train_classifier()
    print(f"Total examples in train_expert: {len(train_expert)}")

    uid_to_subject = {ex["uid"]: ex["subject"] for ex in train_expert}
    uid_to_freq = {}

    for expert_id in expert_ids:
        freq_path = os.path.join(base_path, f"adapter_for_expert_{expert_id}", "training_data_freq.json")
        if not os.path.exists(freq_path):
            print(f"[Warning] {freq_path} not found. Skipping expert {expert_id}.")
            continue
        with open(freq_path, "r") as f:
            uid_to_freq[expert_id] = json.load(f)
    
    return uid_to_subject, uid_to_freq

def build_analysis_dataframe(assignments, uid_to_subject, uid_to_freq_map, expert_id):
    expert_freq_map = uid_to_freq_map.get(expert_id, {})
    data = []
    for uid in assignments:
        subject = uid_to_subject.get(uid)
        if subject is None:
            print(f"[Warning] UID {uid} not found in subject map. Skipping.")
            continue
        freq = expert_freq_map.get(uid, 1)
        data.append({'uid': uid, 'subject': subject, 'freq': freq, 'expert_id': expert_id})
    return pd.DataFrame(data)

def plot_normalized_distributions(df, uid_to_subject, uid_to_freq_map, num_epochs=10, save_path=None):
    """
    Plot normalized exposure of each expert to subjects.

    Normalized exposure:
        [sum of (data point an expert saw from subject s x respective frequency)]
        -----------------------------------------------------
        (total number of samples from subject s x num_epochs)

    Measures how much each expert was exposed to a subject,
    relative to the subject's availability and training duration.
    """

    # Step 1: Count total number of data points per subject across train_full
    all_subjects = [uid_to_subject[uid] for uid in uid_to_subject]
    subject_total_counts = pd.Series(all_subjects).value_counts()  # e.g., {"law": 3000, "math": 1200, ...}

    # Step 2: Compute weighted count per (expert_id, subject) using pre-attached freq column
    weighted_counts = df.groupby(["expert_id", "subject"])["freq"].sum().unstack(fill_value=0)

    expected_counts = subject_total_counts * num_epochs
    normalized = weighted_counts.divide(expected_counts, axis=1)

    col_sums = normalized.sum(axis=0)
    failed = [(s, total) for s, total in col_sums.items() if abs(total - 1.0) > 1e-4]
    if failed:
        print("[ERROR] The following subjects do not sum to 1.0 across experts:")
        for s, val in failed:
            print(f" - {s}: {val:.5f}")
        raise ValueError("Normalization check failed.")
    else:
        print("[OK] All subjects correctly normalized across experts.")

    experts = normalized.index.tolist()
    ncols = 5
    nrows = math.ceil(len(experts) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 6 * nrows), squeeze=False)
    highlight_mask = normalized.eq(normalized.max(axis=0))
    x_max = normalized.max().max() * 1.1

    for idx, expert_id in enumerate(experts):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        subject_values = normalized.loc[expert_id].sort_values()
        subject_names = subject_values.index
        is_max = highlight_mask.loc[expert_id, subject_names]
        bar_colors = ['#FFA500' if flag else '#4682B4' for flag in is_max]
        subject_values.plot(kind='barh', ax=ax, color=bar_colors)
        ax.axvline(1.0 / len(experts), color='black', linestyle='--', linewidth=1)
        ax.set_xlim(0, x_max)

        ax.set_title(f'Normalized Exposure - Expert {expert_id}', fontsize=12)
        ax.set_xlabel('Normalized Exposure', fontsize=10)
        ax.set_ylabel('Subject', fontsize=10)
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', labelsize=8)

    # Hide unused subplots
    for idx in range(len(experts), nrows * ncols):
        row, col = divmod(idx, ncols)
        fig.delaxes(axes[row][col])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=500)
    plt.show()

def plot_subject_distributions(df, save_path=None):
    experts = sorted(df['expert_id'].unique())
    num_experts = len(experts)
    ncols = 3
    nrows = math.ceil(num_experts / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 6 * nrows), squeeze=False)

    for idx, expert_id in enumerate(experts):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        sub_df = df[df['expert_id'] == expert_id]
        sub_df['subject'].value_counts().plot(kind='barh', ax=ax)

        ax.set_title(f'Subject Distribution - Expert {expert_id}', fontsize=12)
        ax.set_xlabel('Num Samples', fontsize=10)
        ax.set_ylabel('Subject', fontsize=10)
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', labelsize=8)

    for idx in range(num_experts, nrows * ncols):
        row, col = divmod(idx, ncols)
        fig.delaxes(axes[row][col])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=500)
    plt.show()

class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    @property
    def data(self):
        return list(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def load_cluster_assignments(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)
    
def get_embeddings(texts, method='sentence-transformer'):
    if method == 'sentence-transformer':
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(texts, convert_to_numpy=True)
    
    elif method == 'tfidf':
        vect = TfidfVectorizer(stop_words='english')
        X    = vect.fit_transform(texts)               # sparse
        n_comp = min(100, X.shape[1] - 1)
        if n_comp <= 0:
            raise ValueError("Too few TF–IDF features (<2) to run TruncatedSVD")
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        return svd.fit_transform(X)
    
    else:
        raise ValueError("Unknown embedding method. Choose 'tfidf' or 'sentence-transformer'.")

def get_expert_assignments(uid_to_freq_map):
    best_assignment = {}
    for expert_id, freqs in uid_to_freq_map.items():
        for uid, freq in freqs.items():
            if uid not in best_assignment or freq > best_assignment[uid][1]:
                best_assignment[uid] = (expert_id, freq)
    return {uid: expert for uid, (expert, _) in best_assignment.items()}

def visualize_expert_assignments_with_tsne_v2(uids, texts, subjects, embeddings, expert_assignments, save_path=None):
    reduced = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
    df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "expert": [expert_assignments.get(uid, -1) for uid in uids],
        "subject": subjects,
        "text": texts
    })

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x="x", y="y", hue="expert", palette="tab10", s=20)
    plt.title("Expert Assignment (Max Freq)")

    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x="x", y="y", hue="subject", palette="tab20", s=20, legend=False)
    plt.title("Ground Truth Subjects")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=400)
    plt.show()

def extract_metadata_from_loader(data_loader):
    texts, uids, subjects, embeddings = [], [], [], []
    for batch in data_loader:
        question = batch["question"]
        answer = batch["mapped_answer"]
        texts.append(f"{question} {answer}")
        uids.append(batch["uid"])
        subjects.append(batch["subject"])
    return texts, uids, subjects


def main(method='sentence-transformer'):
    mmlu = MMLUWrapper()
    num_experts = 10
    expert_ids = list(range(num_experts))
    base_path = "D:/Users/lenovo/Desktop/exp10-lr10-5-temp0.1-epoch10-epoch-9"
    uid_to_subject, uid_to_freq_map = build_uid_maps(mmlu, expert_ids, base_path)
    all_dfs = []

    for expert_id in expert_ids:
        cluster_json_path = f"{base_path}/adapter_for_expert_{expert_id}/training_data_freq.json"
        if not os.path.exists(cluster_json_path):
            print(f"Warning: {cluster_json_path} does not exist. Skipping.")
            continue

        print(f"Processing expert {expert_id}...")
        cluster_assignments = load_cluster_assignments(cluster_json_path)
        df = build_analysis_dataframe(cluster_assignments, uid_to_subject, uid_to_freq_map, expert_id)
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total matched examples across all experts: {len(full_df)}")
    print(f"Total training steps (UID x freq): {full_df['freq'].sum():,}")

    output_dir_name = os.path.basename(base_path.rstrip("/\\"))
    save_dir = os.path.join("cluster_visualization", f"{output_dir_name}_{method}")
    os.makedirs(save_dir, exist_ok=True)

    print("Plotting raw subject distributions...")
    plot_subject_distributions(full_df, save_path=os.path.join(save_dir, "all_expert_subject_distributions.png"))
    
    print("Plotting normalized subject exposures...")
    plot_normalized_distributions(
        full_df,
        uid_to_subject=uid_to_subject,
        uid_to_freq_map=uid_to_freq_map,
        num_epochs=10,
        save_path=os.path.join(save_dir, "normalized_subject_exposures.png")
    )

    raw_data = mmlu.get_train_classifier()
    wrapped = DatasetWrapper(raw_data)
    texts, uids, subjects = [], [], []
    for item in wrapped:
        texts.append(f"{item['question']} {item['mapped_answer']}")
        uids.append(item['uid'])
        subjects.append(item['subject'])

    print(f"Generating embeddings using `{method}`...")
    embeddings = get_embeddings(texts, method=method)

    print("Computing expert assignments...")
    expert_assignments = get_expert_assignments(uid_to_freq_map)

    visualize_expert_assignments_with_tsne_v2(
        uids=uids,
        texts=texts,
        subjects=subjects,
        embeddings=embeddings,
        expert_assignments=expert_assignments,
        save_path=os.path.join(save_dir, "tsne_expert_vs_subject.png")
    )


if __name__ == "__main__":
    main(method='tfidf')