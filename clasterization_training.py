import os

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

EPOCHS_FROM_ORIGIN_MODEL = 30
EPOCHS = EPOCHS_FROM_ORIGIN_MODEL + 60

credentials_data = load_comet_credentials('credentials.json')
COMET_API_KEY, COMET_PROJECT_NAME, COMET_WORKSPACE_NAME = credentials_data['COMET_API_KEY'], credentials_data['COMET_PROJECT_NAME'], credentials_data['COMET_WORKSPACE_NAME']

def train_cluster(datamodule: ImageDatasetLightning, cluster_name: str = 'unknown', load_origin_model: bool = True):
    data_module = datamodule
    data_module.setup()

    if load_origin_model:
        resume_ckpt = f'unet_models/model.pt'
    else:
        resume_ckpt = f'unet_models/model_{cluster_name}.pt'

    model = UNetLightning(lr=LEARNING_RATE, comet_workspace=COMET_WORKSPACE_NAME,
                          comet_project_name=COMET_PROJECT_NAME, comet_api_key=COMET_API_KEY)

    progress_callback = TQDMProgressBar(refresh_rate=1)

    checkpoint_callback = ModelCheckpoint(
        dirpath='unet_models',
        filename='model-{step}-cluster{cluster_id}',
        every_n_epochs=10,
        save_top_k=-1
    )

    comet_logger = CometLogger(
        api_key=COMET_API_KEY,
        project=COMET_PROJECT_NAME,
        workspace=COMET_WORKSPACE_NAME,
        name=f'UNetIP_{cluster_name}'
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

    trainer.save_checkpoint(f'unet_models/model_{cluster_name}.pt')




if __name__ == "__main__":

    target_pth = 'UN/clusters/cluster_3'
    corrupted_pth = 'UN/corrupted_clusters/cluster_3'

    # for target, corrupted in zip(os.listdir(target_pth), os.listdir(corrupted_pth)):
    #     print(f"Training cluster: {target}")
    #     data_module = ImageDatasetLightning(corrupted_dir=os.path.join(corrupted_pth, corrupted),
    #                                         original_dir=os.path.join(target_pth, target),
    #                                         batch_size=8)
    #     train_cluster(data_module, target, load_origin_model=True)

    print(f"Training cluster: {target_pth}")
    data_module = ImageDatasetLightning(corrupted_dir=corrupted_pth,
                                        original_dir=target_pth,
                                        batch_size=8)
    train_cluster(data_module, target_pth, load_origin_model=True)