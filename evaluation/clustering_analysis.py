import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score
from datasets import load_dataset
from data.mmlu.mmludataset import MMLUWrapper

def load_cluster_assignments(json_path):
    with open(json_path, 'r') as f:
        cluster_assignments = json.load(f)
    return cluster_assignments

def build_uid_to_subject_map(mmlu_wrapper):
    train_expert = mmlu_wrapper.get_train_expert()
    uid_to_subject = {example['uid']: example['subject'] for example in train_expert}
    return uid_to_subject

def build_analysis_dataframe(assignments, uid_to_subject):
    data = []
    for uid in assignments:  # ignore cluster value since it's always 0
        subject = uid_to_subject.get(uid)
        if subject is not None:
            data.append({'uid': uid, 'subject': subject})
    return pd.DataFrame(data)

def plot_subject_distribution(df, save_path=None):
    plt.figure(figsize=(12, 8))
    df['subject'].value_counts().plot(kind='barh')
    plt.title('Subject Distribution in Expert Training Set')
    plt.xlabel('Number of Samples')
    plt.ylabel('Subject')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(df, save_path=None):
    pivot = pd.crosstab(df['subject'], df['cluster'])
    plt.figure(figsize=(18, 12))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues")
    plt.title('Subject vs. Cluster Assignment', fontsize=18)
    plt.xlabel('Cluster', fontsize=14)
    plt.ylabel('Subject', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def compute_clustering_metrics(df):
    true_labels = df['subject'].astype('category').cat.codes
    pred_labels = df['cluster']

    ari_score = adjusted_rand_score(true_labels, pred_labels)
    nmi_score = normalized_mutual_info_score(true_labels, pred_labels)

    return ari_score, nmi_score

def main():
    # Adjust path to your actual JSON file
    cluster_json_path = "D:/Users/lenovo/Desktop/low-mi-temp-epoch-9/adapter_for_expert_0/training_data_freq.json"

    print("Loading cluster assignments...")
    cluster_assignments = load_cluster_assignments(cluster_json_path)

    print("Loading MMLU dataset...")
    mmlu = MMLUWrapper()

    print("Building UID to subject map...")
    uid_to_subject = build_uid_to_subject_map(mmlu)

    print("Building analysis dataframe...")
    df = build_analysis_dataframe(cluster_assignments, uid_to_subject)

    print(f"Total matched examples: {len(df)}")

    print("Plotting subject distribution...")
    # plot_confusion_matrix(df, save_path="evaluation/subject_vs_cluster_heatmap.png")
    plot_subject_distribution(df, save_path="evaluation/expert0_subject_distribution.png")


    # print("Computing clustering metrics...")
    # ari, nmi = compute_clustering_metrics(df)
    # print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    # print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

if __name__ == "__main__":
    main()
