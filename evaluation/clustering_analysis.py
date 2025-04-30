import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

import json
import pandas as pd
import matplotlib.pyplot as plt
from data.mmlu.mmludataset import MMLUWrapper
import math

def load_cluster_assignments(json_path):
    with open(json_path, 'r') as f:
        cluster_assignments = json.load(f)
    return cluster_assignments

def build_uid_to_subject_map(mmlu_wrapper):
    train_expert = mmlu_wrapper.get_train_expert()
    uid_to_subject = {example['uid']: example['subject'] for example in train_expert}
    return uid_to_subject

def build_analysis_dataframe(assignments, uid_to_subject, expert_id):
    data = []
    for uid in assignments:
        subject = uid_to_subject.get(uid)
        if subject is not None:
            data.append({'uid': uid, 'subject': subject, 'expert_id': expert_id})
    return pd.DataFrame(data)

def plot_subject_distributions(df, save_path=None):
    experts = sorted(df['expert_id'].unique())
    num_experts = len(experts)

    fig, axes = plt.subplots(nrows=num_experts, ncols=1, figsize=(40, 20))
    if num_experts == 1:
        axes = [axes]

    for ax, expert_id in zip(axes, experts):
        sub_df = df[df['expert_id'] == expert_id]
        sub_df['subject'].value_counts().plot(kind='barh', ax=ax)
        ax.set_title(f'Subject Distribution - Expert {expert_id}')
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Subject')
        ax.tick_params(axis='y', labelsize=5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
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

    # Hide unused subplots (if fewer than nrows * ncols)
    for idx in range(num_experts, nrows * ncols):
        row, col = divmod(idx, ncols)
        fig.delaxes(axes[row][col])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def main():
    mmlu = MMLUWrapper()
    uid_to_subject = build_uid_to_subject_map(mmlu)

    all_dfs = []
    base_path = "D:/Users/lenovo/Desktop/low-mi-temp-epoch-9"

    for expert_id in range(6):
        cluster_json_path = f"{base_path}/adapter_for_expert_{expert_id}/training_data_freq.json"
        if not os.path.exists(cluster_json_path):
            print(f"Warning: {cluster_json_path} does not exist. Skipping.")
            continue

        print(f"Processing expert {expert_id}...")
        cluster_assignments = load_cluster_assignments(cluster_json_path)
        df = build_analysis_dataframe(cluster_assignments, uid_to_subject, expert_id)
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total matched examples across all experts: {len(full_df)}")

    print("Plotting subject distributions for all experts...")
    plot_subject_distributions(full_df, save_path="evaluation/all_expert_subject_distributions.png")

if __name__ == "__main__":
    main()