import comet_ml
import torch
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl
import cv2
import time
import numpy as np



class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 activation: nn.Module, pool_layer:bool = True):
        super(ConvBlock, self).__init__()

        self.activation = activation
        self.pool_layer = pool_layer

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, connecting_layer):
        conv = self.conv1(connecting_layer)
        act = self.activation(conv)
        conv2 = self.conv2(act)
        act2 = self.activation(conv2)

        if self.pool_layer:
            pooled = self.pool(act2)
            return act2, pooled

        return act2

class UpConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: nn.Module):
        super(UpConvBlock, self).__init__()

        self.activation = activation

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

    def forward(self, x, skip):
        x = self.up(x)

        x = torch.cat([x, skip], dim=1)

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        return x



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.ConvBlock1 = ConvBlock(3, 32, 3, 1, 1, nn.ReLU())
        self.ConvBlock2 = ConvBlock(32, 64, 3, 1, 1, nn.ReLU())
        self.ConvBlock3 = ConvBlock(64, 128, 3, 1, 1, nn.ReLU())
        self.ConvBlock4 = ConvBlock(128, 256, 3, 1, 1, nn.ReLU())
        self.ConvBlock5 = ConvBlock(256, 512, 3, 1, 1, nn.ReLU(), pool_layer=False)

        self.UpConvBlock1 = UpConvBlock(512, 256, nn.ReLU())
        self.UpConvBlock2 = UpConvBlock(256, 128, nn.ReLU())
        self.UpConvBlock3 = UpConvBlock(128, 64, nn.ReLU())
        self.UpConvBlock4 = UpConvBlock(64, 32, nn.ReLU())

        self.ConvBlock6 = ConvBlock(32, 32, 3, 1, 1, nn.ReLU(), pool_layer=False)

        self.last_conv = nn.Conv2d(32, 3, 3, padding=1)
        self.output = nn.Sigmoid()

    def forward(self, x):
        c1, p1 = self.ConvBlock1(x)
        c2, p2 = self.ConvBlock2(p1)
        c3, p3 = self.ConvBlock3(p2)
        c4, p4 = self.ConvBlock4(p3)
        c5 = self.ConvBlock5(p4)

        u1 = self.UpConvBlock1(c5, c4)
        u2 = self.UpConvBlock2(u1, c3)
        u3 = self.UpConvBlock3(u2, c2)
        u4 = self.UpConvBlock4(u3, c1)

        cl = self.ConvBlock6(u4)

        out = self.output(self.last_conv(cl))
        return out


class UNetLightning(pl.LightningModule):
    def __init__(self, criterion: nn.Module = nn.CrossEntropyLoss(), lr: float = 1e-3,
                 comet_api_key: str = None, comet_project_name: str = None, comet_workspace: str = None):
        super(UNetLightning, self).__init__()
        self.save_hyperparameters(ignore=['criterion'])
        self.model = UNet()
        self.criterion = criterion
        self.comet_api_key = comet_api_key
        self.comet_project_name = comet_project_name
        self.comet_workspace = comet_workspace
        self.experiment = None

        if self.comet_api_key:
            self.experiment = comet_ml.Experiment(
                api_key=self.comet_api_key,
                project_name=self.comet_project_name,
                workspace=self.comet_workspace
            )
            self.experiment.set_name(f"unet_run_{int(time.time())}")
            try:
                hparams = dict(self.hparams)
            except Exception:
                hparams = {k: v for k, v in self.hparams.items()}
            self.experiment.log_parameters(hparams)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.experiment is not None:
            try:
                self.experiment.log_metric('train_loss', float(loss.detach().cpu().numpy()), step=int(self.global_step))
            except Exception:
                pass
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.experiment is not None and batch_idx == 0:
            try:
                inp = x[0].detach().cpu().numpy()  # (C,H,W)
                pred = y_hat[0].detach().cpu().numpy()

                inp = np.transpose(inp, (1, 2, 0))
                pred = np.transpose(pred, (1, 2, 0))

                def to_uint8(img):
                    img = np.clip(img, 0, 1) if img.max() <= 1.0 else img
                    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                    return img

                inp_img = to_uint8(inp)
                pred_img = to_uint8(pred)

                self.experiment.log_image(inp_img, name="val_input", step=int(self.current_epoch))
                self.experiment.log_image(pred_img, name="val_pred", step=int(self.current_epoch))
            except Exception:
                pass

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)

        if self.experiment is not None:
            try:
                self.experiment.log_metric('test_loss', float(loss.detach().cpu().numpy()), step=int(self.global_step))
            except Exception:
                pass

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

