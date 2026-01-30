import torch
from torchvision import transforms
from PIL import Image
import os

from unet_autoencoder import UNetLightning

CHECKPOINT_PATH = "unet_models"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def apply_inpainting(image, cluster_id: int):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # adjust if needed
        transforms.ToTensor(),
    ])

    inverse_transform = transforms.Compose([
        transforms.ToPILImage()
    ])

    model_path = os.path.join(CHECKPOINT_PATH, f"model_cluster_{str(cluster_id)}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model = UNetLightning.load_from_checkpoint(
        model_path,
        map_location=DEVICE
    )
    model.eval()
    model.to(DEVICE)
    to_pil = transforms.ToPILImage()
    input_tensor = transform(to_pil(image)).unsqueeze(0).to(DEVICE)  # (1, C, H, W)

    with torch.no_grad():
        output = model(input_tensor)

    output_image = inverse_transform(output.squeeze(0).cpu())
    return output_image