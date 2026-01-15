import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class VDSR(nn.Module):
    def __init__(self, depth=20, channels=64):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(1, channels, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(depth - 2):
            layers.append(nn.Conv2d(channels, channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(channels, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def forward(self, x):
        return self.net(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class VDSRLightning(pl.LightningModule):
    def __init__(self, lr=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model = VDSR(depth=20)

        # Metrics (full-image)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        ilr, hr = batch
        residual_gt = hr - ilr
        residual_pred = self(ilr)

        loss = F.mse_loss(residual_pred, residual_gt)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ilr, hr = batch

        with torch.no_grad():
            residual = self(ilr)
            sr = torch.clamp(ilr + residual, 0.0, 1.0)

        if batch_idx == 0:
            self.logger.experiment.log_image(
                sr[0].cpu(),
                name=f"sr_epoch_{self.current_epoch}"
            )
            self.logger.experiment.log_image(
                hr[0].cpu(),
                name=f"hr_epoch_{self.current_epoch}"
            )

        loss = F.mse_loss(sr, hr)
        psnr = self.psnr(sr, hr)
        ssim = self.ssim(sr, hr)

        self.log("val_mse", loss, prog_bar=True, sync_dist=True)
        self.log("val_psnr", psnr, prog_bar=True, sync_dist=True)
        self.log("val_ssim", ssim, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.1
        )

        return [optimizer], [scheduler]