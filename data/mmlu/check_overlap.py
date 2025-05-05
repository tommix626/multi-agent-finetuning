from itertools import combinations
from mmludataset import MMLUWrapper

def check_split_overlap(dataset_dict):
    all_uids = {k: set(v['uid']) for k, v in dataset_dict.items()}
    for a, b in combinations(all_uids.keys(), 2):
        overlap = all_uids[a] & all_uids[b]
        if overlap:
            print(f"Overlap between {a} and {b}: {len(overlap)} entries")
            print("Sample UIDs:", list(overlap)[:5])
        else:
            print(f"No overlap between {a} and {b}")

# Usage
wrapper = MMLUWrapper()
check_split_overlap(wrapper.get_dataset())
