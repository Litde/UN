import torch
from database import ArtDatabase
from torchvision import transforms
import os

def get_dataset():
    db = ArtDatabase(task="inpainting")
    db.download()
    db.make_split(val_size=0.0, test_size=0.2, seed=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # or 256×256 if used in training
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet normalization (ResNet encoder)
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return db.as_torch(split="train", transform=transform)

def save_clusters(directory, dataset, clusters_in, indices_in):
    for cluster_id, image_idx in zip(clusters_in, indices_in):
        cluster_directory = f"cluster_{cluster_id.item()}"
        cluster_directory = os.path.join(directory, cluster_directory)
        if not os.path.exists(cluster_directory):
            os.makedirs(cluster_directory)
        img, _ = dataset[image_idx.item()]
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(cluster_directory, f"image_{image_idx.item()}.png"))

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Loading dataset...")

    torch_dataset = get_dataset()
    print("Loaded dataset with", len(torch_dataset), "samples.")

    ckpt = torch.load("wikiart_latent_features.pt", weights_only=False)

    features = ckpt["features"]
    clusters = ckpt["clusters"]
    indices = ckpt["indices"]

    print("Loaded features and 10 clusters from checkpoint.")

    # Put the desired directory to save clusters
    save_clusters("../UN/clusters", torch_dataset, clusters, indices)
    print("Saved clustered images into respective directories.")
