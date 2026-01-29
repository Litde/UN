import torch
from PIL import Image
from torchvision.transforms import functional as TF
import cv2
from vdsr import VDSRLightning

SCALE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/vdsr/vdsr-000-37.50.ckpt"

def super_resolve(image_path: str):
    model = VDSRLightning.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval().to(DEVICE)

    img = Image.open(image_path).convert("YCbCr")
    y, cb, cr = img.split()

    w, h = y.size
    w -= w % SCALE
    h -= h % SCALE
    y = y.crop((0, 0, w, h))
    cb = cb.crop((0, 0, w, h))
    cr = cr.crop((0, 0, w, h))

    # lr = y.resize((w // SCALE, h // SCALE), Image.BICUBIC)
    # ilr = y.resize((w * SCALE, h * SCALE), Image.BICUBIC)


    w_hr, h_hr = w * SCALE, h * SCALE
    cb = cb.resize((w_hr, h_hr), Image.BICUBIC)
    cr = cr.resize((w_hr, h_hr), Image.BICUBIC)
    lr = y.resize((w, h), Image.BICUBIC)  # low-res same as original
    ilr = lr.resize((w_hr, h_hr), Image.BICUBIC)  # upscale for VDSR
    ilr = TF.to_tensor(ilr).unsqueeze(0).to(DEVICE)

    with torch.no_grad(), torch.cuda.amp.autocast():
        residual = model(ilr)
        sr_y = torch.clamp(ilr + residual, 0.0, 1.0)

    sr_y = TF.to_pil_image(sr_y.squeeze(0).cpu())
    sr_img = Image.merge("YCbCr", (sr_y, cb, cr)).convert("RGB")

    return sr_img
    # sr_img.save(output_path)
    # print(f"Saved super-resolved image to {output_path}")

if __name__ == "__main__":
    input_image_path = "wikiart/0003_12_02.jpg"

    sr_image = super_resolve(input_image_path)
    sr_image.save("super_resolved_image.jpg")
    # cv2.imshow("Result Image", sr_image)