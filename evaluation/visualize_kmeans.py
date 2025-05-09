import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)
from data.mmlu.mmludataset import k_means_clustering, MMLUWrapper   
from torch.utils.data import DataLoader, Dataset

def visualize_clusters(texts, embeddings, cluster_labels, subject_labels, method='pca', save_path=None):
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    reduced = reducer.fit_transform(embeddings)

    df = pd.DataFrame({
        'x': reduced[:, 0],
        'y': reduced[:, 1],
        'cluster': cluster_labels,
        'subject': subject_labels,
        'text': texts
    })

    plt.figure(figsize=(12, 5))

    # Cluster plot
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='tab10', s=20)
    plt.title('KMeans Cluster Assignments')

    # Subject plot
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='x', y='y', hue='subject', palette='tab20', s=20, legend=False)
    plt.title('True Subject Labels')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
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




if __name__ == "__main__":
    mmlu = MMLUWrapper()

    raw_data = mmlu.get_train_classifier()
    wrapped_dataset = DatasetWrapper(raw_data)
    train_loader = DataLoader(wrapped_dataset, batch_size=1)

    clusters, texts, subjects, embeddings, labels = k_means_clustering(
        train_loader,
        k=6,
        method='tfidf',
        return_metadata=True
    )

    visualize_clusters(
        texts=texts,
        embeddings=embeddings,
        cluster_labels=labels,
        subject_labels=subjects,
        method='tsne',
        save_path='kmeans_cluster_vs_subject.png'
    )
