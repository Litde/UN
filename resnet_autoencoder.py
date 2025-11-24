import torch
import torch.nn as nn
from comet_ml.scripts.comet_check import activate_debug
from torch import Tensor
import pytorch_lightning as pl
from dataclasses import dataclass
import cv2
import numpy as np

def prepare_input(image_pth: str, debug:bool = False) -> Tensor:
    img = cv2.imread(image_pth)
    img = cv2.resize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, 3, 256, 256)
    if debug:
        print(f"Prepared input shape: {img.shape}")
    return torch.tensor(img).float()

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

        self.Up1 = UpConvBlock(512, 256, nn.ReLU())
        self.Up2 = UpConvBlock(256, 128, nn.ReLU())
        self.Up3 = UpConvBlock(128, 64, nn.ReLU())
        self.Up4 = UpConvBlock(64, 32, nn.ReLU())

        self.ConvBlock6 = ConvBlock(32, 32, 3, 1, 1, nn.ReLU(), pool_layer=False)

        self.last_conv = nn.Conv2d(32, 3, 3, padding=1)
        self.output = nn.Sigmoid()

    def forward(self, x):
        c1, p1 = self.ConvBlock1(x)
        c2, p2 = self.ConvBlock2(p1)
        c3, p3 = self.ConvBlock3(p2)
        c4, p4 = self.ConvBlock4(p3)
        c5 = self.ConvBlock5(p4)

        u1 = self.Up1(c5, c4)
        u2 = self.Up2(u1, c3)
        u3 = self.Up3(u2, c2)
        u4 = self.Up4(u3, c1)

        cl = self.ConvBlock6(u4)

        out = self.output(self.last_conv(cl))
        return out


class UNetLightning(pl.LightningModule):
    def __init__(self, criterion: nn.Module = nn.CrossEntropyLoss(), lr: float = 1e-3):
        super(UNetLightning, self).__init__()
        self.seve_hyperparameters()
        self.model = UNet()
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

