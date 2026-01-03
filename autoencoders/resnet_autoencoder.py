"""resnet_autoencoder.py

Autoencoder with an inline ResNet encoder + a lightweight U-Net–style decoder.
No external encoder module needed.

Highlights
----------
- Self-contained: defines the ResNet *encoder*, decoder, and AE wrapper here.
- Supports ResNet 18/34/50/101/152 with output_stride 32/16/8.
- Optional multi-scale skips for U-Net–like decoding (C5->...->C1).

Example
-------
>>> import torch
>>> from autoencoders.resnet_autoencoder import ResNetAutoEncoder
>>> ae = ResNetAutoEncoder(name="resnet18", pretrained=True, output_stride=32,
...                        use_skips=True, out_channels=3, base_width=256)
>>> x = torch.randn(2, 3, 256, 256)
>>> y, feats = ae(x)
>>> y.shape
torch.Size([2, 3, 256, 256])
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision

# =============================
# Inline ResNet encoder
# =============================

def _get_weights_arg(name: str, pretrained: bool):
    """Resolve torchvision weights argument across versions."""
    try:  # torchvision >= 0.13 uses weights enums
        if not pretrained:
            return {"weights": None}
        mapping = {
            "resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
            "resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
            "resnet101": torchvision.models.ResNet101_Weights.IMAGENET1K_V1,
            "resnet152": torchvision.models.ResNet152_Weights.IMAGENET1K_V1,
        }
        return {"weights": mapping[name]}
    except Exception:
        # older torchvision fallback
        return {"pretrained": bool(pretrained)}


class ResNetEncoder(nn.Module):
    """TorchVision ResNet as a pure convolutional encoder.

    Parameters
    ----------
    name : {"resnet18","resnet34","resnet50","resnet101","resnet152"}
    pretrained : bool, load ImageNet weights
    output_stride : {32, 16, 8}
        Controls spatial downsampling by altering strides/dilations in late blocks.
    return_stages : sequence of stages to return (subset of C1..C5). If one stage
        is requested, forward returns a Tensor; otherwise a list of Tensors.
    freeze_at : int, optional
        Freeze layers up to this stage inclusive (0=stem, 1=layer1, ..., 4=layer4).
    norm_eval : bool, optional
        If True, set BatchNorm layers to eval (freeze running stats).
    """

    def __init__(
        self,
        name: str = "resnet18",
        pretrained: bool = True,
        output_stride: int = 32,
        return_stages: Sequence[str] = ("C5",),
        freeze_at: int = 0,
        norm_eval: bool = False,
    ) -> None:
        super().__init__()
        assert output_stride in (32, 16, 8), "output_stride must be {32,16,8}"
        self.name = name
        self.output_stride = output_stride
        self.return_stages = tuple(return_stages)

        ctor = getattr(torchvision.models, name)

        is_basic = name in ("resnet18", "resnet34")

        if is_basic:
            base = ctor(**_get_weights_arg(name, pretrained))
            # OS=16: turn off downsampling w layer4[0]
            if output_stride == 16:
                # stem: /4, layer2: /8, layer3: /16, layer4: /16 (stride=1)
                base.layer4[0].conv1.stride = (1, 1)
                if base.layer4[0].downsample is not None:
                    base.layer4[0].downsample[0].stride = (1, 1)
            # OS=8: turn off downsampling in layer3[0] and layer4[0]
            elif output_stride == 8:
                # stem: /4, layer2: /8, layer3: /8, layer4: /8
                base.layer3[0].conv1.stride = (1, 1)
                if base.layer3[0].downsample is not None:
                    base.layer3[0].downsample[0].stride = (1, 1)
                base.layer4[0].conv1.stride = (1, 1)
                if base.layer4[0].downsample is not None:
                    base.layer4[0].downsample[0].stride = (1, 1)
        else:
            rstd = (False, False, False)  # OS=32 (default)
            if output_stride == 16:
                rstd = (False, False, True)  # layer4 with dilation
            elif output_stride == 8:
                rstd = (False, True, True)  # layer3 and layer4 with dilation
            base = ctor(
                **_get_weights_arg(name, pretrained),
                replace_stride_with_dilation=rstd,
            )

        # keep conv backbone; drop avgpool/fc
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)  # C1
        self.layer1 = base.layer1  # C2
        self.layer2 = base.layer2  # C3
        self.layer3 = base.layer3  # C4
        self.layer4 = base.layer4  # C5

        self._maybe_freeze(freeze_at)
        self.norm_eval = norm_eval
        if self.norm_eval:
            self._set_bn_eval()

        self.out_channels: Dict[str, int] = self._infer_out_channels()

    def _infer_out_channels(self) -> Dict[str, int]:
        if self.name in ("resnet18", "resnet34"):
            return {"C1": 64, "C2": 64, "C3": 128, "C4": 256, "C5": 512}
        return {"C1": 64, "C2": 256, "C3": 512, "C4": 1024, "C5": 2048}

    def _maybe_freeze(self, freeze_at: int) -> None:
        stages = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        for i, s in enumerate(stages):
            if i <= freeze_at:
                for p in s.parameters():
                    p.requires_grad_(False)

    def _set_bn_eval(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.requires_grad_(False)

    @torch.no_grad()
    def _feats_shapes(self) -> Dict[str, int]:
        return self.out_channels

    def forward(self, x: torch.Tensor):
        c1 = self.stem(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        feats = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}
        selected = [feats[s] for s in self.return_stages]
        return selected[0] if len(selected) == 1 else selected


# =============================
# Decoder blocks
# =============================
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample x2 + optional skip concat + convs."""
    def __init__(self, in_ch: int, out_ch: int, with_skip: bool = True, mid_ratio: float = 0.5):
        super().__init__()
        self.with_skip = with_skip
        self.mid_ch = int(max(1, in_ch * mid_ratio))

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvBNReLU(in_ch, self.mid_ch, k=3, s=1, p=1)

        self.conv2_noskip = ConvBNReLU(self.mid_ch, out_ch, k=3, s=1, p=1)
        self.conv2_skip   = ConvBNReLU(self.mid_ch * 2, out_ch, k=3, s=1, p=1)

        self.proj_skip = None

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv1(x)  # [B, mid_ch, H, W]

        use_skip = self.with_skip and (skip is not None)
        if use_skip:
            if x.shape[-2:] != skip.shape[-2:]:
                skip = nn.functional.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            if self.proj_skip is None or self.proj_skip.in_channels != skip.shape[1]:
                new = nn.Conv2d(skip.shape[1], self.mid_ch, kernel_size=1, bias=False)

                new = new.to(x.device, dtype=torch.float32)
                self.proj_skip = new
                
                if hasattr(self, "_optimizer_ref") and self._optimizer_ref is not None:
                    self._optimizer_ref.add_param_group({"params": self.proj_skip.parameters()})

            skip = self.proj_skip(skip)

            x = torch.cat([x, skip], dim=1) # [B, 2*mid_ch, H, W]
            x = self.conv2_skip(x)
        else:
            x = self.conv2_noskip(x)
        return x


class UNetStyleDecoder(nn.Module):
    """Top-down decoder that mirrors ResNet scales using skip connections."""
    def __init__(
        self,
        in_ch_top: int,
        skip_channels: Sequence[int],
        out_channels: int = 3,
        base_width: int = 256,
        use_skips: bool = True,
        num_ups: int = 5,
        final_act: str | None = None,
    ) -> None:
        super().__init__()
        self.use_skips = use_skips
        self.num_ups = num_ups

        self.head = ConvBNReLU(in_ch_top, base_width, k=1, s=1, p=0)

        stages: list[nn.Module] = []
        in_ch = base_width
        for _ in range(num_ups):
            stages.append(UpBlock(in_ch, in_ch // 2 if in_ch > 64 else 64, with_skip=use_skips))
            in_ch = in_ch // 2 if in_ch > 64 else 64
        self.stages = nn.ModuleList(stages)

        self.tail = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)
        self.final_act = None
        if final_act == "tanh":
            self.final_act = nn.Tanh()
        elif final_act == "sigmoid":
            self.final_act = nn.Sigmoid()

        self._skip_channels = list(skip_channels)

    def forward(self, top: torch.Tensor, skips: Sequence[torch.Tensor] | None = None) -> torch.Tensor:
        x = self.head(top)
        skips = list(skips) if (skips is not None and self.use_skips) else []
        for i, stage in enumerate(self.stages):
            skip = skips[i] if i < len(skips) else None
            x = stage(x, skip)
        x = self.tail(x)
        if self.final_act is not None:
            x = self.final_act(x)
        return x


# =============================
# Autoencoder wrapper
# =============================
class ResNetAutoEncoder(nn.Module):
    def __init__(
        self,
        name: str = "resnet18",
        pretrained: bool = True,
        output_stride: int = 32,
        return_stages: Sequence[str] = ("C5", "C4", "C3", "C2", "C1"),
        use_skips: bool = True,
        out_channels: int = 3,
        base_width: int = 256,
        num_ups: int | None = None,
        final_act: str | None = None,
        freeze_at: int = 0,
        norm_eval: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = ResNetEncoder(
            name=name,
            pretrained=pretrained,
            output_stride=output_stride,
            return_stages=return_stages,
            freeze_at=freeze_at,
            norm_eval=norm_eval,
        )
        ch_map = self.encoder.out_channels
        top_ch = ch_map[return_stages[0]]
        skip_chs = [ch_map[s] for s in return_stages[1:]]

        if num_ups is None:
            # OS=32 -> top 8x8 -> 5 upsamples; OS=16 -> 16x16 -> 4; OS=8 -> 32x32 -> 3
            num_ups = {32: 5, 16: 4, 8: 3}[output_stride]

        self.decoder = UNetStyleDecoder(
            in_ch_top=top_ch,
            skip_channels=skip_chs,
            out_channels=out_channels,
            base_width=base_width,
            use_skips=use_skips,
            num_ups=num_ups,
            final_act=final_act,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
        feats = self.encoder(x)
        if isinstance(feats, torch.Tensor):
            top, skips = feats, []
        else:
            top, skips = feats[0], feats[1:]
        y = self.decoder(top, skips)
        return y, feats


__all__ = [
    "ResNetEncoder",
    "ConvBNReLU",
    "UpBlock",
    "UNetStyleDecoder",
    "ResNetAutoEncoder",
]
