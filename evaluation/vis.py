import os
import sys
import json
import math
import pandas as pd
import matplotlib.pyplot as plt

# If needed: manually set the path to your repo
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

from data.mmlu.mmludataset import MMLUWrapper

def load_cluster_assignments(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

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
            print(f"[Warning] UID {uid} not found in subject map.")
            continue
        freq = expert_freq_map.get(uid, 1)
        data.append({"uid": uid, "subject": subject, "freq": freq, "expert_id": expert_id})
    return pd.DataFrame(data)

def plot_subject_distributions(df, save_path=None):
    experts = sorted(df["expert_id"].unique())
    ncols = 3
    nrows = math.ceil(len(experts) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 6 * nrows), squeeze=False)

    for idx, expert_id in enumerate(experts):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        sub_df = df[df["expert_id"] == expert_id]
        sub_df["subject"].value_counts().plot(kind="barh", ax=ax)
        ax.set_title(f"Subject Distribution - Expert {expert_id}")
        ax.set_xlabel("Num Samples")
        ax.set_ylabel("Subject")

    for idx in range(len(experts), nrows * ncols):
        row, col = divmod(idx, ncols)
        fig.delaxes(axes[row][col])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=500)
    plt.show()

def plot_normalized_distributions(df, uid_to_subject, num_epochs=17, save_path=None):
    all_subjects = [uid_to_subject[uid] for uid in uid_to_subject]
    subject_total_counts = pd.Series(all_subjects).value_counts()

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
    ncols = 3
    nrows = math.ceil(len(experts) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 6 * nrows), squeeze=False)

    for idx, expert_id in enumerate(experts):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        normalized.loc[expert_id].sort_values().plot(kind="barh", ax=ax)
        ax.set_title(f"Normalized Exposure - Expert {expert_id}")
        ax.set_xlabel("Normalized Exposure")
        ax.set_ylabel("Subject")

    for idx in range(len(experts), nrows * ncols):
        row, col = divmod(idx, ncols)
        fig.delaxes(axes[row][col])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=500)
    plt.show()

def main():
    mmlu = MMLUWrapper()
    base_path = "D:/Users/lenovo/Desktop/new-data-split-epoch-17"
    expert_ids = list(range(6))

    uid_to_subject, uid_to_freq_map = build_uid_maps(mmlu, expert_ids, base_path)

    all_dfs = []
    for expert_id in expert_ids:
        path = f"{base_path}/adapter_for_expert_{expert_id}/training_data_freq.json"
        if not os.path.exists(path):
            print(f"[Warning] Missing data for expert {expert_id}")
            continue
        print(f"Processing expert {expert_id}...")
        assignments = load_cluster_assignments(path)
        df = build_analysis_dataframe(assignments, uid_to_subject, uid_to_freq_map, expert_id)
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"✅ Total matched examples (rows): {len(full_df)}")
    print(f"✅ Total training steps (UID × freq): {full_df['freq'].sum():,}")

    os.makedirs("evaluation", exist_ok=True)

    print("📊 Plotting raw subject distributions...")
    plot_subject_distributions(full_df, save_path="evaluation/all_expert_subject_distributions.png")

    print("📊 Plotting normalized subject exposures...")
    plot_normalized_distributions(
        full_df,
        uid_to_subject=uid_to_subject,
        num_epochs=18,
        save_path="evaluation/normalized_subject_exposures.png"
    )

if __name__ == "__main__":
    main()
