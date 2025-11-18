import os

import cv2
from comet_ml import start, login, Experiment
from comet_ml.integration.pytorch import watch

from dataset import ImageMaskDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm




class BasicInpainting(nn.Module):
    def __init__(
        self,
        encoder_channels=(32, 64, 128),
        latent_dim=512,
        num_inpainting_layers=2,
        use_hardswish=True,
        super_res_scale=2,
    ):
        super(BasicInpainting, self).__init__()
        self.latent_dim = latent_dim
        self.encoder_channels = list(encoder_channels)
        self.num_inpainting_layers = num_inpainting_layers
        self.use_hardswish = use_hardswish
        self.super_res_scale = super_res_scale

        act = lambda: nn.Hardswish(True) if use_hardswish else nn.ReLU(True)

        # build encoder: 3 -> *encoder_channels -> latent_dim
        enc_layers = []
        in_ch = 3
        for ch in self.encoder_channels:
            enc_layers += [nn.Conv2d(in_ch, ch, 4, 2, 1), act()]
            in_ch = ch
        # final conv to latent_dim if needed
        if in_ch != latent_dim:
            enc_layers += [nn.Conv2d(in_ch, latent_dim, 4, 2, 1), act()]
        self.encoder_module = nn.Sequential(*enc_layers)

        # inpainting module: repeated convs at latent_dim
        inp_layers = []
        for _ in range(self.num_inpainting_layers):
            inp_layers += [nn.Conv2d(latent_dim, latent_dim, 3, 1, 1), act()]
        self.inpainting_module = nn.Sequential(*inp_layers)

        # build decoder: latent_dim -> reversed encoder channels -> 3
        dec_layers = []
        rev_channels = list(reversed(self.encoder_channels))
        in_ch = latent_dim
        for ch in rev_channels:
            dec_layers += [nn.ConvTranspose2d(in_ch, ch, 4, 2, 1), act()]
            in_ch = ch
        dec_layers += [nn.ConvTranspose2d(in_ch, 3, 4, 2, 1), nn.Sigmoid()]
        self.decoder_module = nn.Sequential(*dec_layers)

        # super resolution module (simple)
        self.super_res_module = nn.Sequential(
            nn.Upsample(scale_factor=self.super_res_scale, mode="bilinear", align_corners=False),
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

        self._to_tensor = transforms.ToTensor()

    def forward(self, x, mask=None):
        if mask is not None:
            x_masked = x * mask
        else:
            x_masked = x

        latent = self.encoder_module(x_masked)
        latent_filled = self.inpainting_module(latent)
        out = self.decoder_module(latent_filled)
        out_sr = self.super_res_module(out)
        return out, out_sr

    # \- pomocnicze metody wejściowe i predict --\ (pozostawić/zaimportować z poprzedniej wersji)
    def _ensure_tensor(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.float()
        if isinstance(arr, Image.Image):
            return self._to_tensor(arr).float()
        if isinstance(arr, np.ndarray):
            a = arr
            if a.dtype == np.uint8:
                a = a.astype(np.float32) / 255.0
            if a.ndim == 2:
                a = np.expand_dims(a, -1)
            if a.ndim == 3:
                a = np.transpose(a, (2, 0, 1))
            return torch.from_numpy(a).float()
        raise TypeError("Unsupported input type for _ensure_tensor")

    def _prepare_batch(self, img, mask=None, device=None):
        img_t = self._ensure_tensor(img) if img is not None else None
        if img_t is None:
            raise ValueError("img cannot be None")
        if img_t.dim() == 2:
            img_t = img_t.unsqueeze(0)
        if img_t.dim() == 3:
            img_t = img_t.unsqueeze(0)
        mask_t = None
        if mask is not None:
            mask_t = self._ensure_tensor(mask)
            if mask_t.dim() == 2:
                mask_t = mask_t.unsqueeze(0)
            if mask_t.dim() == 3:
                mask_t = mask_t.unsqueeze(0) if mask_t.shape[0] != img_t.shape[0] else mask_t
        if device is not None:
            img_t = img_t.to(device)
            if mask_t is not None:
                mask_t = mask_t.to(device)
        mask_3c = None
        if mask_t is not None:
            if mask_t.dim() == 4 and mask_t.shape[1] == 1 and img_t.shape[1] == 3:
                mask_3c = mask_t.repeat(1, 3, 1, 1)
            else:
                mask_3c = mask_t
        return img_t, mask_t, mask_3c

    def train_one_epoch(self, dataloader, optimizer, criterion, device):
        self.train()
        total_loss = 0.0
        for batch in tqdm(dataloader, desc="train", leave=False):
            img, mask = batch
            if not isinstance(img, torch.Tensor) or not isinstance(mask, torch.Tensor):
                img_t, mask_t, mask_3c = self._prepare_batch(img, mask, device=device)
            else:
                img_t = img.to(device)
                mask_t = mask.to(device)
                if mask_t.dim() == 3:
                    mask_t = mask_t.unsqueeze(1)
                mask_3c = mask_t.repeat(1, 3, 1, 1)
            corrupted = img_t * (1 - mask_3c)
            optimizer.zero_grad()
            output, _ = self(corrupted, mask_3c)
            loss = criterion(output * mask_3c, img_t * mask_3c)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(dataloader))
        return avg_loss

    def evaluate(self, dataloader, criterion, device):
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="eval", leave=False):
                img, mask = batch
                if not isinstance(img, torch.Tensor) or not isinstance(mask, torch.Tensor):
                    img_t, mask_t, mask_3c = self._prepare_batch(img, mask, device=device)
                else:
                    img_t = img.to(device)
                    mask_t = mask.to(device)
                    if mask_t.dim() == 3:
                        mask_t = mask_t.unsqueeze(1)
                    mask_3c = mask_t.repeat(1, 3, 1, 1)
                corrupted = img_t * (1 - mask_3c)
                output, _ = self(corrupted, mask_3c)
                loss = criterion(output * mask_3c, img_t * mask_3c)
                total_loss += loss.item()
        avg_loss = total_loss / max(1, len(dataloader))
        return avg_loss

    def predict(self, image, mask=None, device=None, show=False, return_sr=False):
        dev = device if device is not None else next(self.parameters()).device
        self.to(dev)
        self.eval()
        img_t, mask_t, mask_3c = self._prepare_batch(image, mask, device=dev)
        with torch.no_grad():
            out, out_sr = self(img_t, mask_3c)
        key_out = out_sr if return_sr else out
        img_out = key_out[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        img_out = (img_out * 255.0).astype("uint8")
        if img_out.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_out
        if show:
            cv2.imshow("inpainted", img_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return (img_bgr, out_sr) if return_sr else img_bgr

    def load_pretrained_weights(self, path):
        self.load_state_dict(torch.load(path))

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

def import_params(params: dict) -> dict:
    """Importuje parametry modelu z formatu słownikowego Optuna do formatu akceptowanego przez BasicInpainting."""
    base_chan = params.get("base_channels", 32)
    depth = params.get("enc_depth", 3)
    encoder_channels = [base_chan * (2 ** i) for i in range(depth)]
    latent_dim = params.get("latent_dim", 512)
    num_inpainting_layers = params.get("num_inpainting_layers", 2)
    use_hardswish = params.get("use_hardswish", False)

    return {
        "encoder_channels": tuple(encoder_channels),
        "latent_dim": latent_dim,
        "num_inpainting_layers": num_inpainting_layers,
        "use_hardswish": use_hardswish,
    }

def train_model():
    pass  # implementacja treningu modelu tutaj


if __name__ == '__main__':
    pass

