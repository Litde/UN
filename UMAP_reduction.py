import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datasets import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import umap
import umap.plot
import os
import cv2
from tqdm import tqdm

class DataSet:
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def prepare_data(data_dir, target='style'):
    files = os.listdir(data_dir)
    original_map = {}
    data = []
    style_lbl = []
    genre_lbl = []

    for f in tqdm(files):
        file_name = os.path.splitext(f)[0]  # '0000.jpg' -> '0000'
        tab = file_name.split("_")
        num = int(tab[0])
        style = int(tab[1])
        genre = int(tab[2])
        original_map[int(num)] = f
        path = os.path.join(data_dir, f)
        img = cv2.imread(path)
        img = cv2.resize(img, (256, 256))

        data.append(img)
        style_lbl.append(style)
        genre_lbl.append(genre)

        if num >= 1000:
            break
    if target == 'style':
        return DataSet(data, style_lbl)
    else:
        return DataSet(data, genre_lbl)

def reduce_dimensions(data, target, n_neighbors=50, min_dist=0.25, n_components=2, metric='euclidean'):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    embedding = reducer.fit_transform(data)
    umap.plot.points(reducer, labels=target, theme='fire')

    return embedding

def plot_embedding(embedding, target, output_file='umap_plot.png'):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=target, cmap='Spectral', s=5)
    plt.colorbar(scatter)
    plt.title('UMAP projection of the dataset', fontsize=15)
    plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    data_dir = "wikiart"
    dataset = prepare_data(data_dir, target='genre')

    # Flatten images for UMAP
    flattened_data = [img.flatten() for img in dataset.data]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(flattened_data)
    # Reduce dimensions
    # embedding = reduce_dimensions(flattened_data, np.array(dataset.target))
    # plot_embedding(embedding, np.array(dataset.target))

    # Plot embedding
    # plot_embedding(embedding, dataset.target)

    # Create subplots for different parameter settings
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    # Different parameter configurations
    param_configs = [
        {'n_neighbors': 5, 'min_dist': 0.1},
        {'n_neighbors': 15, 'min_dist': 0.1},
        {'n_neighbors': 50, 'min_dist': 0.1},
        {'n_neighbors': 15, 'min_dist': 0.0},
        {'n_neighbors': 15, 'min_dist': 0.5},
        {'n_neighbors': 15, 'min_dist': 0.99}
    ]

    # Apply UMAP with different parameters
    for idx, params in enumerate(param_configs):
        reducer = umap.UMAP(random_state=42, **params)
        embedding = reducer.fit_transform(X_scaled)

        ax = axes[idx]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=dataset.target,
                             cmap='tab20', s=30, alpha=0.8)
        ax.set_title(f"n_neighbors={params['n_neighbors']}, "
                     f"min_dist={params['min_dist']}")
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("umap_parameter_comparison.png")
    plt.show()