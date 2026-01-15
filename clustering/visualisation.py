import torch
from database import ArtDatabase
from torchvision import transforms
import umap
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def show_cluster_images(cluster_id, clusters, dataset, max_images=16, cols=4,):
    idxs = [i for i, c in enumerate(clusters) if c == cluster_id]
    idxs = idxs[:max_images]

    images = []
    for i in idxs:
        img, _ = dataset[i]
        images.append(img)

    grid = make_grid(images, nrow=cols, normalize=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title(f"Cluster {cluster_id} ({len(idxs)} samples shown)")
    plt.show()

def show_UMAP(X, clusters):
    umap_model = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42,
    )

    X_2d = umap_model.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=clusters,
        cmap="tab10",
        s=10,
        alpha=0.8,
    )

    plt.colorbar(scatter, label="Cluster ID")
    plt.title("Latent space clustering (UMAP)")
    plt.show()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Loading dataset...")
    db = ArtDatabase(task="inpainting")
    db.download()
    db.make_split(val_size=0.0, test_size=0.2, seed=42)

    dataset = db.get_train()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # or 256×256 if used in training
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet normalization (ResNet encoder)
            std=[0.229, 0.224, 0.225],
        ),
    ])

    torch_dataset = db.as_torch(split="train", transform=transform)
    print("Loaded dataset with", len(torch_dataset), "samples.")
    ckpt = torch.load("wikiart_latent_features.pt", weights_only=False)

    features = ckpt["features"]
    clusters = ckpt["clusters"]
    indices = ckpt["indices"]

    print("Loaded features and 10 clusters from checkpoint.")

    X = features.numpy()

    # show_UMAP(X, clusters)
    show_cluster_images(0, clusters, torch_dataset)
    show_cluster_images(1, clusters, torch_dataset)
    show_cluster_images(2, clusters, torch_dataset)
    show_cluster_images(3, clusters, torch_dataset)
    show_cluster_images(4, clusters, torch_dataset)
    show_cluster_images(5, clusters, torch_dataset)
    show_cluster_images(6, clusters, torch_dataset)
    show_cluster_images(7, clusters, torch_dataset)
    show_cluster_images(8, clusters, torch_dataset)
    show_cluster_images(9, clusters, torch_dataset)