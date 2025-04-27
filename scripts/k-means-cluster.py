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

import sys
sys.path.insert(0, "/Users/ruoxiliu/Desktop/NLP-final/multi-agent-finetuning/data/mmlu")

from mmluwrapper import MMLUWrapper

print("1")
"""
Example Usage:
    # Cluster using TF-IDF on two text files into 5 clusters and save results:
    python cluster_text.py \
        --files ./data/mmlu/train/train.txt \
        --method tfidf \
        --output ./data/mmlu/clustered_train/tfidf \
        --k 5

    # Cluster using SentenceTransformer embeddings:
    python cluster_text.py \
        --files ./data/mmlu/train/train.txt \
        --method sentence-transformer \
        --output ./data/mmlu/clustered_train/sentence-transformer \
        --k 8
"""

def tfidf(sentences, k):
    """
    Args:
        sentences: List of sentences to cluster
        k: Number of clusters
    
    Returns:
        list: List of k clusters, each a list of datapoints
    """
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Transform sentences to TF-IDF features
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Apply PCA for dimensionality reduction (optional but can improve clustering)
    pca = PCA(n_components=min(100, tfidf_matrix.shape[1] - 1))
    reduced_features = pca.fit_transform(tfidf_matrix.toarray())
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(reduced_features)
    
    # Organize sentences by cluster
    clustered_sentences = [[] for _ in range(k)]
    for i, cluster_id in enumerate(clusters):
        clustered_sentences[cluster_id].append(sentences[i])
    
    return clustered_sentences

def sentence_transformer(sentences, k):
    """
    Args:
        sentences: List of sentences to cluster
        k: Number of clusters
    
    Returns:
        list: List of k clusters, each a list of datapoints
    """
    # Load pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings
    embeddings = model.encode(sentences)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Organize sentences by cluster
    clustered_sentences = [[] for _ in range(k)]
    for i, cluster_id in enumerate(clusters):
        clustered_sentences[cluster_id].append(sentences[i])
    
    return clustered_sentences

def main():
    # Command line argument parser
    parser = argparse.ArgumentParser(description='Cluster text data using k-means')
    parser.add_argument('--files', nargs='+', required=True, help='List of input files')
    parser.add_argument('--method', choices=['tfidf', 'sentence-transformer'], required=True, 
                        help='Clustering method: tfidf or sentence-transformer')
    parser.add_argument('--output', required=True, help='Output directory for clustered data')
    parser.add_argument('--k', type=int, default=5, help='Number of clusters (default: 5)')
    
    args = parser.parse_args()
    
    # Read and combine all input files
    all_sentences = []
    for file_path in args.files:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Strip newlines and filter empty lines
            lines = [line.strip() for line in lines if line.strip()]
            all_sentences.extend(lines)
    
    print(f"Read {len(all_sentences)} sentences from {len(args.files)} files")
    
    # Apply clustering based on selected method
    if args.method == 'tfidf':
        clustered_data = tfidf(all_sentences, args.k)
    else:  # sentence-transformer
        clustered_data = sentence_transformer(all_sentences, args.k)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Write clusters to separate files
    for i, cluster in enumerate(clustered_data):
        output_file = os.path.join(args.output, f'cluster_{i}.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in cluster:
                f.write(f"{sentence}\n")
        print(f"Cluster {i} has {len(cluster)} sentences - written to {output_file}")

if __name__ == "__main__":
    main()