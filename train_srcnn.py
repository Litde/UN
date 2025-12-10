import comet_ml
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning import Trainer
from srcnn import SRCNNLitModel
from super_resolution_dataset import ImageDatasetLightning
from utils import load_comet_credentials
import torch
import os
import glob

EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

credentials_data = load_comet_credentials('credentials.json')
COMET_API_KEY, COMET_PROJECT_NAME, COMET_WORKSPACE_NAME = credentials_data['COMET_API_KEY'], credentials_data['COMET_PROJECT_NAME'], credentials_data['COMET_WORKSPACE_NAME']

def _get_latest_checkpoint(dirpath: str):
    patterns = [os.path.join(dirpath, '*.ckpt'), os.path.join(dirpath, '*.pt')]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

def train_model():
    data_module = ImageDatasetLightning(image_dir='wikiart_super_res', batch_size=BATCH_SIZE)
    data_module.setup()

    latest_ckpt = _get_latest_checkpoint('srcnn_models')
    if latest_ckpt:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        resume_ckpt = latest_ckpt
    else:
        print("No checkpoint found, training from scratch.")
        resume_ckpt = None

    model = SRCNNLitModel(learning_rate=LEARNING_RATE, comet_workspace=COMET_WORKSPACE_NAME,
                          comet_project_name=COMET_PROJECT_NAME, comet_api_key=COMET_API_KEY)

    progress_callback = TQDMProgressBar(refresh_rate=1)

    checkpoint_callback = ModelCheckpoint(
        dirpath='srcnn_models',
        filename='model-{step}',
        every_n_epochs=1,
        save_top_k=-1
    )

    comet_logger = CometLogger(
        api_key=COMET_API_KEY,
        project=COMET_PROJECT_NAME,
        workspace=COMET_WORKSPACE_NAME,
    )

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = 1 if torch.cuda.is_available() else None
    # devices = 1

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

    trainer.save_checkpoint('srcnn_models/model.pt')

def test_model():
    data_module = ImageDatasetLightning(image_dir='wikiart_super_res', batch_size=BATCH_SIZE)
    data_module.setup()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SRCNNLitModel.load_from_checkpoint('unet_models/model.pt',
                                               map_location=device,
                                               comet_api_key=COMET_API_KEY,
                                               comet_workspace=COMET_WORKSPACE_NAME,
                                               comet_project_name=COMET_PROJECT_NAME)
    trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1 if torch.cuda.is_available() else None)

    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    train_model()