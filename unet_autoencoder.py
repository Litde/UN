# file: `unet_autoencoder.py` (updated _log_images_to_comet)
from datetime import datetime

import cv2
from PIL import Image
import comet_ml
import torch
import torch.nn as nn
from pytorch_msssim import ssim
import pytorch_lightning as pl

# enable reduced-precision matmul for Tensor Cores when CUDA is available
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')

L1 = nn.L1Loss()
SSIM = ssim


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=False):
        super().__init__()
        self.pool = pool
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        if pool:
            self.down = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.block(x)
        return (x, self.down(x)) if self.pool else x


class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch*2, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cb1 = ConvBlock(3,   32, pool=True)
        self.cb2 = ConvBlock(32,  64, pool=True)
        self.cb3 = ConvBlock(64, 128, pool=True)
        self.cb4 = ConvBlock(128, 256, pool=True)
        self.cb5 = ConvBlock(256, 512, pool=False)

        self.up1 = UpConvBlock(512, 256)
        self.up2 = UpConvBlock(256, 128)
        self.up3 = UpConvBlock(128,  64)
        self.up4 = UpConvBlock(64,   32)

        self.out = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        c1, p1 = self.cb1(x)
        c2, p2 = self.cb2(p1)
        c3, p3 = self.cb3(p2)
        c4, p4 = self.cb4(p3)
        c5     = self.cb5(p4)

        x = self.up1(c5, c4)
        x = self.up2(x, c3)
        x = self.up3(x, c2)
        x = self.up4(x, c1)
        return self.out(x)


class UNetLightning(pl.LightningModule):
    def __init__(self, lr=1e-3, comet_api_key=None, comet_project_name=None, comet_workspace=None):
        super().__init__()
        self.save_hyperparameters()

        self.model = UNet()
        self.lr = lr

        self.l1 = nn.L1Loss()
        self.ssim_weight = 0.15

        # Do not create a separate comet_ml.Experiment here.
        # Use the Trainer/CometLogger integration (self.logger.experiment) instead.
        self._example_batch = None  # will hold one example for epoch-level image logging

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, pred, target):
        ssim_val = SSIM(pred, target, data_range=1.0)
        return 0.85 * self.l1(pred, target) + self.ssim_weight * (1 - ssim_val)

    def training_step(self, batch, _):
        x, y = batch
        device = self.device
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        y_hat = self(x)

        loss = self.compute_loss(y_hat, y)

        # log training loss every step (CometLogger will pick this up)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # removed manual self.experiment.log_metric(...) so metrics are only logged via self.log

        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch
        device = self.device
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        y_hat = self(x)

        loss = self.compute_loss(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)

        # keep per-test-image logging only if needed; here we avoid per-step image logging
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        device = self.device
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        y_hat = self(x)

        loss = self.compute_loss(y_hat, y)
        # aggregate validation loss per-epoch; no per-step val metric required
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

        # store first sample of the first batch (detached to CPU) for epoch-level image logging
        if batch_idx == 0:
            try:
                self._example_batch = (
                    x[0].detach().cpu().clone(),
                    y_hat[0].detach().cpu().clone(),
                    y[0].detach().cpu().clone(),
                )
            except Exception:
                self._example_batch = None

        return loss

    def on_validation_epoch_end(self):
        # Log images once per validation epoch using the Trainer's logger (CometLogger)
        if not self._example_batch:
            return

        # Get comet experiment from the logger if available
        exp = None
        logger = getattr(self, "logger", None)
        if logger is not None:
            # CometLogger exposes the Experiment via .experiment
            exp = getattr(logger, "experiment", None)

        if exp is None:
            # nothing to do if no comet experiment present
            self._example_batch = None
            return

        x, pred, y = self._example_batch

        def to_uint8(t):
            img = (t.clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
            return img

        img = to_uint8(x[:3])
        out = to_uint8(pred[:3])
        tgt = to_uint8(y[:3])

        epoch = int(self.current_epoch)
        step = int(self.global_step)

        # log images once per epoch; use epoch as step/metadata for clarity
        try:
            exp.log_image(img, name=f"val_input_epoch_{epoch}", step=epoch, metadata={"epoch": epoch})
            exp.log_image(out, name=f"val_output_epoch_{epoch}", step=epoch, metadata={"epoch": epoch})
            exp.log_image(tgt, name=f"val_target_epoch_{epoch}", step=epoch, metadata={"epoch": epoch})
        except Exception:
            pass

        # clear stored sample
        self._example_batch = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
