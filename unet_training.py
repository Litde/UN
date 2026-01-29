from datetime import datetime

import comet_ml
import cv2

from unet_autoencoder import UNetLightning
from dataset import ImageDatasetLightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CometLogger

from utils import load_comet_credentials
import torch
import os
import glob

# enable reduced-precision matmul for Tensor Cores when CUDA is available
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')

EPOCHS = 30
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
    data_module = ImageDatasetLightning(corrupted_dir='output/images', original_dir='wikiart', batch_size=BATCH_SIZE)
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

def test_model():
    data_module = ImageDatasetLightning(corrupted_dir='output/images', original_dir='wikiart', batch_size=BATCH_SIZE)
    data_module.setup()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # find the latest checkpoint in the folder
    latest_ckpt = _get_latest_checkpoint('unet_models')
    if latest_ckpt is None:
        # fallback to a default file if present
        fallback = 'unet_models/model.pt'
        if os.path.exists(fallback):
            latest_ckpt = fallback

    if latest_ckpt is None:
        print("No checkpoint found under 'unet_models'. Please train the model first.")
        return

    print(f"Loading checkpoint: {latest_ckpt}")

    # load model with checkpoint weights onto the correct device
    model = UNetLightning.load_from_checkpoint(
        latest_ckpt,
        map_location=device,
        comet_api_key=COMET_API_KEY,
        comet_workspace=COMET_WORKSPACE_NAME,
        comet_project_name=COMET_PROJECT_NAME,
    )
    model.to(device)
    model.eval()

    # create a Trainer consistent with available hardware
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = 1 if torch.cuda.is_available() else None
    trainer = Trainer(accelerator=accelerator, devices=devices)

    # run Lightning's test loop (will call model.test_step etc.)
    print("Running trainer.test(...)")
    trainer.test(model, datamodule=data_module)

    # perform a quick inference on one batch from the test dataloader for visual inspection
    try:
        test_loader = data_module.test_dataloader()
        sample_batch = next(iter(test_loader))
    except Exception as e:
        print(f"Could not fetch a batch from test_dataloader: {e}")
        return

    corrupted_imgs, original_imgs = sample_batch  # tensors in shape (B, C, H, W), floats in [0,1]
    corrupted_imgs = corrupted_imgs.to(device, non_blocking=True)

    with torch.no_grad():
        recon_imgs = model(corrupted_imgs)

    # prepare single example for display/save
    def to_bgr_uint8(tensor):
        # tensor: (C,H,W) float [0,1]
        img = tensor.clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
        img = (img * 255.0).astype('uint8')
        # convert RGB -> BGR for OpenCV
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    idx = 0
    orig_np = to_bgr_uint8(original_imgs[idx])
    corrupted_np = to_bgr_uint8(corrupted_imgs[idx].cpu())
    recon_np = to_bgr_uint8(recon_imgs[idx])

    # try to show using OpenCV; if that fails (headless), save to disk
    try:
        cv2.imshow('Original', orig_np)
        cv2.imshow('Corrupted', corrupted_np)
        cv2.imshow('Reconstructed', recon_np)
        print("Press any key in the image window to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        out_dir = 'unet_models/test_outputs'
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, 'original.png'), orig_np)
        cv2.imwrite(os.path.join(out_dir, 'corrupted.png'), corrupted_np)
        cv2.imwrite(os.path.join(out_dir, 'reconstructed.png'), recon_np)
        print(f"Display failed ({e}). Saved sample images to {out_dir}")

if __name__ == "__main__":
    train_model()
    # test_model()