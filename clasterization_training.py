from database import ArtDatabase
from unet_autoencoder import UNetLightning
from unet_training import _get_latest_checkpoint, LEARNING_RATE
from utils import load_comet_credentials
from dataset import ImageDatasetLightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CometLogger
import torch
from datetime import datetime
from torchvision import transforms


EPOCHS = 10

credentials_data = load_comet_credentials('credentials.json')
COMET_API_KEY, COMET_PROJECT_NAME, COMET_WORKSPACE_NAME = credentials_data['COMET_API_KEY'], credentials_data['COMET_PROJECT_NAME'], credentials_data['COMET_WORKSPACE_NAME']

def train_cluster(datamodule: ImageDatasetLightning):
    data_module = datamodule
    data_module.setup()

    latest_ckpt = _get_latest_checkpoint('unet_models')
    if latest_ckpt:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        resume_ckpt = latest_ckpt
    else:
        print("No checkpoint found, training from scratch.")
        resume_ckpt = None

    model = UNetLightning(lr=LEARNING_RATE, comet_workspace=COMET_WORKSPACE_NAME,
                          comet_project_name=COMET_PROJECT_NAME, comet_api_key=COMET_API_KEY)

    progress_callback = TQDMProgressBar(refresh_rate=1)

    checkpoint_callback = ModelCheckpoint(
        dirpath='unet_models',
        filename='model-{step}',
        every_n_epochs=1,
        save_top_k=-1
    )

    comet_logger = CometLogger(
        api_key=COMET_API_KEY,
        project=COMET_PROJECT_NAME,
        workspace=COMET_WORKSPACE_NAME,
        name=f'UNetIP_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = 1 if torch.cuda.is_available() else None

    trainer = Trainer(
        max_epochs=EPOCHS,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, progress_callback],
        enable_progress_bar=True,
        logger=comet_logger,
        log_every_n_steps=1,
    )

    if resume_ckpt:
        trainer.fit(model, data_module, ckpt_path=resume_ckpt)
    else:
        trainer.fit(model, data_module)

    trainer.save_checkpoint('unet_models/model.pt')

def get_original_dataset() -> tuple:
    db = ArtDatabase(task="inpainting")
    db.download()
    db.make_split(val_size=0.0, test_size=0.2, seed=42)

    dataset = db.get_train()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    torch_dataset = db.as_torch(split="train", transform=transform)
    print("Loaded dataset with", len(torch_dataset), "samples.")
    ckpt = torch.load("clustering/wikiart_latent_features.pt", weights_only=False)

    features = ckpt["features"]
    clusters = ckpt["clusters"]
    indices = ckpt["indices"]

    return torch_dataset, features, clusters, indices


if __name__ == "__main__":
    origin_dataset, features, clusters, indices = get_original_dataset()

    image_datasets = []
    for cluster_id in range(10):
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        selected_dataset = torch.utils.data.Subset(origin_dataset, [indices[i] for i in cluster_indices])
        ds = ImageDatasetLightning()
        image_datasets.append(selected_dataset)
