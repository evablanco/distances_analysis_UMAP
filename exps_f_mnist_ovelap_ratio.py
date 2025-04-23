import pandas as pd
import numpy as np
import umap
from keras.datasets import fashion_mnist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


def load_fmnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    X_train = np.reshape(x_train, (x_train.shape[0], -1)).astype(np.float32)
    X_test = np.reshape(x_test, (x_test.shape[0], -1)).astype(np.float32)
    data = np.concatenate((X_train, X_test), 0)
    labels = np.concatenate((y_train, y_test))
    return data, labels


def calculate_overlap_ratio(data, labels, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    indices = nbrs.kneighbors(data, return_distance=False)

    total_overlap = 0
    n_samples = len(labels)

    for i in range(n_samples):
        neighbor_labels = labels[indices[i]]
        unique_classes, counts = np.unique(neighbor_labels, return_counts=True)
        max_class_fraction = max(counts) / n_neighbors
        total_overlap += 1 - max_class_fraction

    overlap_ratio = total_overlap / n_samples
    return round(overlap_ratio, 4)


def calculate_silhouette_score(data, labels):
    try:
        score = silhouette_score(data, labels)
        return round(score, 4)
    except ValueError as e:
        print(f"Silhouette Score calculation failed: {e}")
        return None


def run_umap_and_calculate_overlap(data, labels, metrics, neighbors, min_dist_vals, output_csv):
    results = []

    for metric in metrics:
        for n_neighbors in neighbors:
            for min_dist in min_dist_vals:
                reducer = umap.UMAP(n_neighbors=n_neighbors, metric=metric, min_dist=min_dist)
                embeddings = reducer.fit_transform(data)

                overlap_ratio = calculate_overlap_ratio(embeddings, labels, n_neighbors)
                silhouette = calculate_silhouette_score(embeddings, labels)
                print(f"Processing: Metric={metric}, n_neighbors={n_neighbors}, min_dist={min_dist}: silhouette_score={silhouette}")

                results.append({
                    "Metric": metric,
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "Overlap_Ratio": overlap_ratio,
                    "Silhouette_Score": silhouette
                })
    return results


def start_exps_fmnist():
    # Parameters
    metrics = ['euclidean', 'manhattan', 'canberra', 'cosine', 'braycurtis']
    neighbors = [15, 30, 50, 100]
    min_dist_vals = [0.01, 0.05, 0.1, 0.5, 1]
    output_csv = "umap_fmnist_silhouette_results.csv"
    data, labels = load_fmnist()
    # Run UMAP and calculate overlap ratios
    results = run_umap_and_calculate_overlap(data, labels, metrics, neighbors, min_dist_vals, output_csv)
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")



