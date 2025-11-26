from datetime import datetime

import cv2
from PIL import Image
import comet_ml
import torch
import torch.nn as nn
from pytorch_msssim import ssim
import pytorch_lightning as pl


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

        self.experiment = (
            comet_ml.Experiment(
                api_key=comet_api_key,
                project_name=comet_project_name,
                workspace=comet_workspace,
                auto_output_logging=False,
                auto_param_logging=False,
                auto_metric_logging=False,
            )
            if comet_api_key else None
        )

        if self.experiment is not None:
            self.experiment.set_name(f"UNet_Autoencoder_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}")

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, pred, target):
        ssim_val = SSIM(pred, target, data_range=1.0)
        return 0.85 * self.l1(pred, target) + self.ssim_weight * (1 - ssim_val)

    def training_step(self, batch, _):
        x, y = batch

        y_hat = self(x)

        loss = self.compute_loss(y_hat, y)

        self.log("train_loss", loss, on_step=True, prog_bar=True)

        if self.experiment:
            self.experiment.log_metric("train_loss", loss.item(), step=self.global_step)

        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.compute_loss(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)

        if self.experiment and batch_idx == 0:
            self._log_images_to_comet(x[0], y_hat[0], y[0])

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.compute_loss(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        if self.experiment and batch_idx == 0:
            self._log_images_to_comet(x[0], y_hat[0], y[0])

        return loss

    def _log_images_to_comet(self, x, pred, y):
        if not self.experiment:
            return

        img = (x[:3]).byte().permute(1, 2, 0).numpy()
        out = (pred[:3]).byte().permute(1, 2, 0).numpy()
        tgt = (y[:3]).byte().permute(1, 2, 0).numpy()

        cv2.imshow("img", img)
        cv2.imshow("out", out)
        cv2.imshow("tgt", tgt)

        self.experiment.log_image(img, name="val_input", step=self.current_epoch)
        self.experiment.log_image(out, name="val_output", step=self.current_epoch)
        self.experiment.log_image(tgt, name="val_target", step=self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


