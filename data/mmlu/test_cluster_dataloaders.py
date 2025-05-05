import torch
from transformers import AutoModelForCausalLM
from mmludataset import get_clustered_dataloaders

def test_clustered_dataloaders():
    model_name = "EleutherAI/gpt-neo-125m"  # Or your desired model
    batch_size = 8
    num_clusters = 3
    method = 'tfidf'  # or 'sentence-transformer'

    print("Fetching clustered dataloaders...")
    cluster_loaders, dev_loader, test_loader = get_clustered_dataloaders(
        model_name=model_name,
        batch_size=batch_size,
        k=num_clusters,
        method=method
    )

    print("\n=== Cluster Summary ===")
    for idx, loader in enumerate(cluster_loaders):
        total = sum(1 for _ in loader)
        print(f"Cluster {idx}: {total * batch_size} examples (approx)")

    print("\n=== Dev/Test Batches ===")
    print(f"Dev batches: {sum(1 for _ in dev_loader)}")
    print(f"Test batches: {sum(1 for _ in test_loader)}")

if __name__ == "__main__":
    test_clustered_dataloaders()