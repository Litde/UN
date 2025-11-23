import cv2

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np


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

        act = (lambda: nn.Hardswish(True)) if use_hardswish else (lambda: nn.ReLU(True))

        # encoder
        enc_layers = []
        in_ch = 3
        for ch in self.encoder_channels:
            enc_layers += [nn.Conv2d(in_ch, ch, 4, 2, 1), act()]
            in_ch = ch
        if in_ch != latent_dim:
            enc_layers += [nn.Conv2d(in_ch, latent_dim, 4, 2, 1), act()]
        self.encoder = nn.Sequential(*enc_layers)

        # inpainting
        inp_layers = []
        for _ in range(self.num_inpainting_layers):
            inp_layers += [nn.Conv2d(latent_dim, latent_dim, 3, 1, 1), act()]
        self.inpainting = nn.Sequential(*inp_layers)

        # decoder
        dec_layers = []
        rev = list(reversed(self.encoder_channels))
        in_ch = latent_dim
        for ch in rev:
            dec_layers += [nn.ConvTranspose2d(in_ch, ch, 4, 2, 1), act()]
            in_ch = ch
        dec_layers += [nn.ConvTranspose2d(in_ch, 3, 4, 2, 1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

        # super-res (optional)
        self.super_res = nn.Sequential(
            nn.Upsample(scale_factor=self.super_res_scale, mode="bilinear", align_corners=False),
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

        self._to_tensor = transforms.ToTensor()

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        z = self.encoder(x)
        z = self.inpainting(z)
        out = self.decoder(z)
        out_sr = self.super_res(out)
        return out, out_sr

    def _ensure_tensor(self, arr):
        # Accept torch.Tensor, PIL.Image, numpy.ndarray
        if isinstance(arr, torch.Tensor):
            return arr.float()
        if isinstance(arr, Image.Image):
            return self._to_tensor(arr).float()
        if isinstance(arr, np.ndarray):
            a = arr.astype(np.float32)
            if a.max() > 2.0:
                a = a / 255.0
            if a.ndim == 2:
                a = np.expand_dims(a, -1)
            if a.ndim == 3:
                a = np.transpose(a, (2, 0, 1))
            return torch.from_numpy(a).float()
        raise TypeError("Unsupported input type")

    def _prepare_batch(self, img, mask=None, device=None):
        img_t = self._ensure_tensor(img)
        if img_t.dim() == 2:
            img_t = img_t.unsqueeze(0)
        if img_t.dim() == 3:
            img_t = img_t.unsqueeze(0)
        mask_t = None
        if mask is not None:
            mask_t = self._ensure_tensor(mask)
            if mask_t.dim() == 2:
                mask_t = mask_t.unsqueeze(0)
            if mask_t.dim() == 3 and mask_t.shape[0] != img_t.shape[0]:
                mask_t = mask_t.unsqueeze(0)
        if device is not None:
            img_t = img_t.to(device)
            if mask_t is not None:
                mask_t = mask_t.to(device)
        if mask_t is not None and mask_t.dim() == 4 and mask_t.shape[1] == 1 and img_t.shape[1] == 3:
            mask_3c = mask_t.repeat(1, 3, 1, 1)
        else:
            mask_3c = mask_t
        return img_t, mask_t, mask_3c

    def predict(self, image, mask=None, device=None, show=False, return_sr=False):
        # accept file paths for convenience
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        if isinstance(mask, str):
            if mask.lower().endswith('.npy'):
                arr = np.load(mask)
                mask = Image.fromarray((arr * 255).astype('uint8'))
            else:
                try:
                    mask = Image.open(mask).convert('L')
                except Exception:
                    mask = None
        dev = device if device is not None else next(self.parameters()).device
        self.to(dev)
        self.eval()
        img_t, mask_t, mask_3c = self._prepare_batch(image, mask, device=dev)
        with torch.no_grad():
            out, out_sr = self(img_t, mask_3c)
        out_key = out_sr if return_sr else out
        img_out = out_key[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        img_out = (img_out * 255).astype('uint8')
        if img_out.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_out
        if show:
            cv2.imshow('pred', img_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return (img_bgr, out_sr) if return_sr else img_bgr

    def load_pretrained_weights(self, path):
        import torch
        try:
            sd = torch.load(path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f'Could not load checkpoint: {e}')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        # strip "module." if present
        if isinstance(sd, dict):
            new = {}
            for k, v in sd.items():
                nk = k[7:] if k.startswith('module.') else k
                new[nk] = v
            sd = new
        self.load_state_dict(sd)


def import_params(file_path: str) -> dict:
    import json
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    params = data.get('params', data if isinstance(data, dict) else {})
    base = int(params.get('base_channels', 32))
    depth = int(params.get('enc_depth', 3))
    encoder_channels = tuple(base * (2 ** i) for i in range(depth))
    return {
        'encoder_channels': encoder_channels,
        'latent_dim': int(params.get('latent_dim', 512)),
        'num_inpainting_layers': int(params.get('num_inpainting_layers', 2)),
        'use_hardswish': bool(params.get('use_hardswish', False)),
    }


if __name__ == '__main__':
    pass
