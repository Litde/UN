import comet_ml
import torch
from torch import nn
import math
import pytorch_lightning as pl
import numpy as np
import cv2

import utils

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')

class SRCNN(nn.Module):
    def __init__(self) -> None:
        super(SRCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True)
        )

        self.map = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True)
        )

        self.reconstruction = nn.Conv2d(32, 1, (5, 5), (1, 1), (2, 2))

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)

class SRCNNLitModel(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-4, momentum = 0.9, weight_decay = 1e-4, nesterov = False, comet_api_key=None, comet_project_name=None, comet_workspace=None) -> None:
        super().__init__()
        self.model = SRCNN()
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        # store an example batch for logging validation images
        # initialize to None to avoid AttributeError when Trainer calls hooks
        self._example_batch = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    @torch.no_grad()
    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        self.log('test_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)

        # Save one example batch (detached and moved to cpu) for logging in on_validation_epoch_end
        if self._example_batch is None:
            try:
                self._example_batch = (
                    inputs.detach().cpu(),
                    outputs.detach().cpu(),
                    targets.detach().cpu()
                )
            except Exception:
                # if anything goes wrong, ensure we don't raise inside the training loop
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
            # ensure tensor is on cpu and detached
            t = t.detach().cpu()
            # handle batched tensor (N, C, H, W) -> (N, H, W, C)
            if t.dim() == 4:
                return (t.clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(0, 2, 3, 1).numpy()
            # single image tensor (C, H, W) -> (H, W, C)
            return (t.clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()

        def process_images(t):
            # print(t.shape)
            sr_y_image = utils.tensor2image(t, False, True)
            sr_y_image = sr_y_image.astype(np.float32) / 255.0
            return sr_y_image

        # img = to_uint8(x[:3])
        # out = to_uint8(pred[:3])
        # tgt = to_uint8(y[:3])

        img = process_images(x[:1])
        out = process_images(pred[:1])
        tgt = process_images(y[:1])

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD([{"params": self.model.features.parameters()},
                           {"params": self.model.map.parameters()},
                           {"params": self.model.reconstruction.parameters(), "lr": self.learning_rate * 0.1}],
                          lr=self.learning_rate,
                          momentum=self.momentum,
                          weight_decay=self.weight_decay,
                          nesterov=self.nesterov)
        return optimizer