import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

import json
import pandas as pd
import matplotlib.pyplot as plt
from data.mmlu.mmludataset import MMLUWrapper
import math
from collections import defaultdict

def load_cluster_assignments(json_path):
    with open(json_path, 'r') as f:
        cluster_assignments = json.load(f)
    return cluster_assignments

def build_uid_maps_fixed(mmlu_wrapper, expert_ids, base_path):
    """Build mappings between UIDs, subjects, and expert frequencies"""
    try:
        dataset = mmlu_wrapper.get_analysis_dataset()
        if not dataset:
            raise ValueError("Analysis dataset is empty")
            
        # Build subject map and keep track of all UIDs
        uid_to_subject = {}
        all_uids_in_dataset = set()
        for ex in dataset:
            uid_to_subject[ex['uid']] = ex['subject']
            all_uids_in_dataset.add(ex['uid'])
            
        print(f"Total UIDs in analysis dataset: {len(all_uids_in_dataset)}")
        
        # Build frequency map and track matching UIDs
        uid_to_freq_map = defaultdict(list)
        matched_uids = set()
        
        for expert_id in expert_ids:
            freq_path = os.path.join(base_path, f"adapter_for_expert_{expert_id}", "training_data_freq.json")
            if not os.path.exists(freq_path):
                print(f"Warning: {freq_path} not found. Skipping expert {expert_id}.")
                continue
                
            with open(freq_path) as f:
                expert_freqs = json.load(f)
                if not isinstance(expert_freqs, dict):
                    print(f"Warning: Invalid format in {freq_path}. Expected dict, got {type(expert_freqs)}")
                    continue
                    
            for uid, freq in expert_freqs.items():
                if uid in uid_to_subject:  # Only include UIDs that exist in our subject map
                    uid_to_freq_map[uid].append((expert_id, freq))
                    matched_uids.add(uid)
        
        print(f"Total UIDs in frequency files: {len(uid_to_freq_map)}")
        print(f"Matched UIDs between frequency and subject data: {len(matched_uids)}")
        
        # Calculate and print mismatch statistics
        unmatched_in_freq = set(uid_to_freq_map.keys()) - matched_uids
        unmatched_in_subject = all_uids_in_dataset - matched_uids
        
        print(f"\nData matching statistics:")
        print(f"- UIDs only in frequency data: {len(unmatched_in_freq)}")
        print(f"- UIDs only in subject data: {len(unmatched_in_subject)}")
        print(f"- Successfully matched UIDs: {len(matched_uids)}")
        
        if not matched_uids:
            raise ValueError("No matching UIDs found between frequency and subject data")
            
        return uid_to_subject, uid_to_freq_map
        
    except Exception as e:
        print(f"Error building UID maps: {str(e)}")
        raise

def build_analysis_dataframe_fixed(uid_to_subject, uid_to_freq_map):
    """Build analysis DataFrame with better mismatch handling"""
    rows = []
    missing_subjects = 0
    invalid_freqs = 0
    
    for uid, exposures in uid_to_freq_map.items():
        subject = uid_to_subject.get(uid)
        if subject is None:
            missing_subjects += 1
            continue
            
        for expert_id, freq in exposures:
            try:
                freq_val = float(freq)
                if freq_val <= 0:
                    invalid_freqs += 1
                    continue
                    
                rows.append({
                    'uid': uid,
                    'subject': subject,
                    'freq': freq_val,
                    'expert_id': int(expert_id)
                })
            except (ValueError, TypeError):
                invalid_freqs += 1
                continue
    
    print(f"\nData quality statistics:")
    print(f"- UIDs missing subject info: {missing_subjects}")
    print(f"- Entries with invalid frequencies: {invalid_freqs}")
    print(f"- Valid entries found: {len(rows)}")
    
    if not rows:
        raise ValueError("No valid data found to build DataFrame. Possible causes:\n"
                        "1. No overlap between frequency UIDs and subject UIDs\n"
                        "2. All frequencies are zero or invalid\n"
                        "3. Expert IDs are missing or invalid")
        
    df = pd.DataFrame(rows)
    
    # Verify required columns exist
    required_columns = ['uid', 'subject', 'freq', 'expert_id']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    return df



def plot_normalized_distributions(df, uid_to_subject, uid_to_freq_map, num_epochs=10, save_path=None):
    """
    Plot normalized exposure of each expert to subjects.
    
    Fixed normalization calculation:
    1. Calculate total exposure per (expert, subject)
    2. Calculate total possible exposure per subject (count * num_epochs)
    3. Normalize by dividing (1) by (2)
    """
    try:
        # Calculate total samples per subject
        subject_counts = pd.Series(uid_to_subject).value_counts()
        
        # Calculate total weighted exposure per (expert, subject)
        exposure_sums = df.groupby(['expert_id', 'subject'])['freq'].sum().unstack(fill_value=0)
        
        # Normalize by dividing by (subject_count * num_epochs)
        normalized = exposure_sums.div(subject_counts * num_epochs)
        
        # Verify normalization sums to ~1 per subject across experts
        col_sums = normalized.sum(axis=0)
        tolerance = 0.01  # Allow 1% tolerance for floating point errors
        failed = [(s, total) for s, total in col_sums.items() 
                 if abs(total - 1.0) > tolerance]
        
        if failed:
            print("[Warning] Some subjects don't sum to 1.0 across experts:")
            for s, total in failed:
                print(f" - {s}: {total:.5f}")
        
        # Plotting
        experts = sorted(normalized.index.tolist())
        ncols = 3
        nrows = math.ceil(len(experts) / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                               figsize=(20, 6 * nrows), 
                               squeeze=False)

        for idx, expert_id in enumerate(experts):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            
            # Sort by normalized exposure value
            expert_data = normalized.loc[expert_id].sort_values()
            expert_data.plot(kind='barh', ax=ax)

            ax.set_title(f'Expert {expert_id}', fontsize=12)
            ax.set_xlabel('Normalized Exposure', fontsize=10)
            ax.set_ylabel('Subject', fontsize=10)
            ax.tick_params(axis='y', labelsize=7)
            ax.tick_params(axis='x', labelsize=8)
            ax.set_xlim(0, 1)  # Ensure consistent scale

        # Hide unused subplots
        for idx in range(len(experts), nrows * ncols):
            row, col = divmod(idx, ncols)
            fig.delaxes(axes[row][col])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Error plotting normalized distributions: {str(e)}")
        raise


    
def plot_subject_distributions(df, save_path=None):
    """Plot subject distributions with better error handling"""
    try:
        # Verify dataframe has required columns
        required_columns = ['expert_id', 'subject']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

        # Get unique experts and handle empty case
        experts = df['expert_id'].unique()
        if len(experts) == 0:
            raise ValueError("No experts found in DataFrame")
        experts = sorted(experts)

        # Setup plot grid
        ncols = 3
        nrows = math.ceil(len(experts) / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                               figsize=(20, 6 * nrows), 
                               squeeze=False)

        for idx, expert_id in enumerate(experts):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]

            # Filter and plot data
            sub_df = df[df['expert_id'] == expert_id]
            if len(sub_df) == 0:
                print(f"Warning: No data for expert {expert_id}")
                continue
                
            sub_df['subject'].value_counts().plot(kind='barh', ax=ax)

            ax.set_title(f'Expert {expert_id}', fontsize=12)
            ax.set_xlabel('Num Samples', fontsize=10)
            ax.set_ylabel('Subject', fontsize=10)
            ax.tick_params(axis='y', labelsize=7)
            ax.tick_params(axis='x', labelsize=8)

        # Hide unused subplots
        for idx in range(len(experts), nrows * ncols):
            row, col = divmod(idx, ncols)
            fig.delaxes(axes[row][col])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Error plotting subject distributions: {str(e)}")
        raise


def main():
    mmlu = MMLUWrapper()
    expert_ids = list(range(6))
    base_path = "D:/Users/lenovo/Desktop/low-mi-temp-epoch-9"

    try:
        # Step 1: Build mappings with detailed diagnostics
        uid_to_subject, uid_to_freq_map = build_uid_maps_fixed(mmlu, expert_ids, base_path)

        # Step 2: Build DataFrame with better error reporting
        df = build_analysis_dataframe_fixed(uid_to_subject, uid_to_freq_map)
        print(f"\nDataFrame successfully created with {len(df)} rows")

        os.makedirs("evaluation", exist_ok=True)

        # Step 3: Plot data
        print("\nPlotting raw subject distributions...")
        plot_subject_distributions(df, save_path="evaluation/all_expert_subject_distributions.png")

        print("\nPlotting normalized subject exposures...")
        plot_normalized_distributions(
            df,
            uid_to_subject=uid_to_subject,
            uid_to_freq_map=uid_to_freq_map,
            num_epochs=10,
            save_path="evaluation/normalized_subject_exposures.png"
        )

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        print("\nDebugging suggestions:")
        print("1. Check if the base path contains the expected expert adapter folders")
        print("2. Verify the training_data_freq.json files exist and contain valid data")
        print("3. Check if the UID formats match between the frequency files and MMLU dataset")
        print("4. Examine the data matching statistics printed above")
        raise


if __name__ == "__main__":
    main()