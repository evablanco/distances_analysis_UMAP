import sklearn
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import umap
from matplotlib.colors import ListedColormap


def set_parameters():
    # Define datasets and UMAP parameters, TO-DO: set in args.
    n_samples = 500
    metrics_name = ['Euclidean', 'Manhattan', 'Canberra', 'Cosine', 'Bray-Curtis']
    metrics = ['euclidean', 'manhattan', 'canberra', 'cosine', 'braycurtis']
    datasets_names = ['Two Lines', 'Two Circles', 'Two Moons']
    umap_params = {'n_neighbors': 15, 'min_dist': 0.1}
    return n_samples, metrics_name, metrics, datasets_names, umap_params

# Generate synthetic datasets
def generate_line(n_samples, offset=0, noise=10):
    x = np.arange(n_samples)
    y = x + np.random.random(n_samples) * noise + offset
    return x, y

def generate_lines_dataset(n_samples, num_lines):
    x, y = [], []
    for i in range(num_lines):
        x_coords, y_coords = generate_line(n_samples, offset=40 * i)
        x.extend(np.column_stack((x_coords, y_coords)))
        y.extend([i] * n_samples)
    return np.array(x), np.array(y)


def start_visualization_toy_exps():
    n_samples, metrics_name, metrics, datasets_names, umap_params = set_parameters()
    x_lines, y_lines = generate_lines_dataset(150, 2)
    x_circles, y_circles = datasets.make_circles(n_samples=n_samples, factor=0.2)
    x_moons, y_moons = datasets.make_moons(n_samples=n_samples)

    datasets_data = [(x_lines, y_lines), (x_circles, y_circles), (x_moons, y_moons)]

    # Initialize the plot grid with an additional column for the original datasets
    fig, axes = plt.subplots(3, 6, figsize=(18, 9), constrained_layout=True)
    custom_cmap = ListedColormap(['#80BD9E', '#336B87'])

    for i, (data, labels) in enumerate(datasets_data):
        # Plot the original dataset in the first column
        ax = axes[i, 0]
        scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap=custom_cmap, s=10, alpha=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:  # Add title to the first column
            ax.set_title("Original Data", fontsize=16)

        # Add the UMAP embeddings for each distance metric
        for j, (metric, metric_name) in enumerate(zip(metrics, metrics_name)):
            embedding = umap.UMAP(n_neighbors=umap_params['n_neighbors'],
                                  min_dist=umap_params['min_dist'],
                                  metric=metric).fit_transform(data)
            ax = axes[i, j + 1]
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=custom_cmap, s=10, alpha=0.8)
            ax.set_xticks([])
            ax.set_yticks([])

            # Add titles for the metrics
            if i == 0:  # Top row: distance metric names
                ax.set_title(metric_name, fontsize=16)

        # Add dataset names to the first column in vertical alignment
        axes[i, 0].annotate(datasets_names[i], xy=(-0.6, 0.5), xycoords='axes fraction',
                            fontsize=16, rotation=90, ha='center', va='center')

    # Save the figure
    plt.savefig("./results/umap_embeddings_grid_with_titles.png", dpi=300)
    plt.show()
