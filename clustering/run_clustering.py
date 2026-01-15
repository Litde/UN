import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.cluster import KMeans

from autoencoders.resnet_autoencoder import ResNetAutoEncoder
from database import ArtDatabase
from tqdm import tqdm

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

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

    loader = DataLoader(
        torch_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = ResNetAutoEncoder(
        name="resnet18",        # MUST match training
        pretrained=False,
        output_stride=32,
    )

    ckpt = torch.load("../runs/ae_5pct/ae_last.pt", map_location="cpu")

    state_dict = ckpt["state_dict"]

    # Remove "module." if trained with DataParallel
    clean_state = {}
    # for k, v in state_dict.items():
    #     if k.startswith("module."):
    #         k = k[len("module."):]
    #     clean_state[k] = v
    for k, v in state_dict.items():
        if k.endswith(".proj_skip.weight") and k.startswith("decoder.stages."):
            continue
        clean_state[k] = v

    model.load_state_dict(clean_state)
    model.to(device)
    model.eval()

    features_all = []
    indices_all = []

    with torch.no_grad():
        idx = 0
        for images, labels in tqdm(loader, desc="Extracting features"):
            bsz = images.size(0)

            images = images.to(device)
            _, feats = model(images)

            c5 = feats[0]
            pooled = c5.mean(dim=(2, 3))  # [B, C]

            features_all.append(pooled.cpu())
            indices_all.extend(range(idx, idx + bsz))

            idx += bsz

    features_all = torch.cat(features_all, dim=0)
    print("Feature matrix:", features_all.shape)

    kmeans = KMeans(n_clusters=10, random_state=42)
    clusters = kmeans.fit_predict(features_all.numpy())

    torch.save(
        {
            "features": features_all,  # [N, D] torch.Tensor
            "clusters": clusters,  # [N] numpy array or list
            "indices": indices_all,  # [N] list of dataset indices
        },
        "wikiart_latent_features.pt"
    )

    # print("Cluster labels:", clusters[:10])
