import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from pytorch_msssim import ssim
import torch.nn as nn
from unet_autoencoder import UNetLightning


L1 = nn.L1Loss()
SSIM = ssim


THRESHOLD = 0.003


CHECKPOINT_PATH = "unet_models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_to_tensor = transforms.ToTensor()
_to_pil = transforms.ToPILImage()

def inpaint_image_file(image_path: str, output_path: str, model):
    image = Image.open(image_path).convert("RGB")

    # Build input/target tensors consistently: [1, C, H, W], float in [0,1]
    x = _to_tensor(image).unsqueeze(0).to(DEVICE)
    target = x  # target is the original (uncorrupted) image here

    model.eval()
    with torch.no_grad():
        pred = model(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]

        # Ensure shape/device match for loss
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        pred = pred.to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0)

        def compute_loss(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            # p,t: [N, C, H, W] in [0,1]
            ssim_weight = 0.15
            ssim_val = SSIM(p, t, data_range=1.0)
            return 0.85 * L1(p, t) + ssim_weight * (1.0 - ssim_val)

        loss = compute_loss(pred, target)


    # Save image (convert from [1,C,H,W] -> [C,H,W] on CPU)
    if loss.item() < THRESHOLD:
        out = pred.squeeze(0).detach().cpu()
        inpainted_image = _to_pil(out)
        print(f"Inpainting {os.path.basename(image_path)} - Loss: {loss.item():.6f}")
        inpainted_image.save(output_path)

def load_model_for_cluster(cluster_id: int):
    model_path = os.path.join(CHECKPOINT_PATH, f"model_cluster_{str(cluster_id)}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model = UNetLightning.load_from_checkpoint(
        model_path,
        map_location=DEVICE
    )
    model.to(DEVICE)
    model.eval()
    return model

def inpaint_images_in_directory(input_dir: str, cluster_id: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    model = load_model_for_cluster(cluster_id)
    print("Loaded model for cluster", cluster_id)

    for filename in tqdm(os.listdir(input_dir), mininterval=0.5, miniters=50, leave=False):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            inpaint_image_file(input_path, output_path, model)

if __name__ == "__main__":
    # cluster_id = 2
    for cluster in tqdm(range(0,10)):
        input_directory = f"UN/corrupted_clusters/cluster_{cluster}"
        output_directory = f"UN/inpainted_clusters_filtered/cluster_{cluster}"

        inpaint_images_in_directory(input_directory, cluster, output_directory)
