import comet_ml
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import Trainer

from train_srcnn import _get_latest_checkpoint
from utils import load_comet_credentials
from vdsr import VDSRLightning
from wikiart_data_module import WikiArtDataModule
from pytorch_lightning.callbacks import ModelCheckpoint

def train_model(comet_api_key, comet_project_name, comet_workspace_name, epochs=10, resume=True):
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/vdsr",
        monitor="val_psnr",
        mode="max",  # higher PSNR is better
        save_top_k=1,  # keep best only
        #save_last=True,  # also save last.ckpt
        every_n_epochs=1,
        filename="vdsr-{epoch:03d}-{val_psnr:.2f}",
        auto_insert_metric_name=False,
    )

    comet_logger = CometLogger(
        api_key=comet_api_key,
        project=comet_project_name,
        workspace=comet_workspace_name,
    )

    model = VDSRLightning(lr=0.1)
    data = WikiArtDataModule()

    trainer = Trainer(
        max_epochs=epochs,
        logger=comet_logger,
        gradient_clip_val=0.4,   # matches paper behavior
        accelerator="gpu",
        devices=1,
        precision="16-mixed",  # AMP
        enable_progress_bar=True,
        log_every_n_steps=10,
        enable_model_summary=True,
        callbacks=[checkpoint_callback],
    )

    if resume:
        trainer.fit(
            model,
            datamodule=data,
            ckpt_path="lightning_logs/version_0/checkpoints/vdsr-012-28.41.ckpt"
        )
    else:
        trainer.fit(model, data)

if __name__ == "__main__":
    credentials_data = load_comet_credentials('credentials.json')
    COMET_API_KEY, COMET_PROJECT_NAME, COMET_WORKSPACE_NAME = credentials_data['COMET_API_KEY'], credentials_data[
        'COMET_PROJECT_NAME'], credentials_data['COMET_WORKSPACE_NAME']

    train_model(COMET_API_KEY, COMET_PROJECT_NAME, COMET_WORKSPACE_NAME, epochs=1, resume=False)