import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import argparse
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torchvision import transforms as T
import cv2

from autoencoders.resnet_autoencoder import ResNetEncoder
from clustering.predict_cluster import predict_cluster
from hole_generator.holes_generator import ImageHoleGenerator
from applying_inpainting import apply_inpainting
from super_resolution.infer_vdsr import super_resolve

ENCODER_WEIGHTS_PATH = "runs/resnet_ae_os16/encoder_best.pt"
BACKBONE = "resnet18"
OUTPUT_STRIDE = 16
IMAGE_SIZE = 256

_encoder_instance = None
_device_instance = None

def _get_or_load_encoder_model():
    global _encoder_instance, _device_instance
    if _encoder_instance is not None:
        return _encoder_instance, _device_instance

    _device_instance = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = ResNetEncoder(
        name=BACKBONE,
        pretrained=False,
        output_stride=OUTPUT_STRIDE,
        return_stages=("C5", "C4", "C3", "C2", "C1"),
    ).to(_device_instance)

    weights_path = Path(ENCODER_WEIGHTS_PATH)
    if not weights_path.exists():
        raise FileNotFoundError(f"Błąd: Nie znaleziono wag enkodera pod ścieżką '{weights_path}'")

    encoder.load_state_dict(torch.load(weights_path, map_location=_device_instance))
    encoder.eval()
    
    _encoder_instance = encoder
    return _encoder_instance, _device_instance


def process_image(image_path: Union[str, Path]) -> torch.Tensor:
    encoder, device = _get_or_load_encoder_model()

    transform = T.Compose([
        T.Resize(IMAGE_SIZE),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono obrazu wejściowego pod ścieżką '{image_path}'")
        raise

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(img_tensor)

    bottleneck_feature = features[0]
    print(f"Kształt wyekstrahowanej cechy (bottleneck): {bottleneck_feature.shape}")

    return bottleneck_feature


def run_restoration_pipeline(image_path: Union[str, Path]) -> str:
    print("--- Starting Image Restoration Pipeline ---")
    
    try:
        # Predict cluster from the original image
        print(f"[1/3] Processing original image for clustering: {image_path}")
        features = process_image(image_path)
        cluster = predict_cluster(features.cpu())
        print(f"      -> Predicted Cluster ID: {cluster}")

        # Load the image object to pass to subsequent functions
        damaged_image = Image.open(image_path).convert("RGB")

        # Inpaint the damaged image using the cluster-specific model
        print(f"[2/3] Inpainting damaged image...")
        restored_image = apply_inpainting(damaged_image, cluster_id=cluster)
        print(f"      -> Inpainting complete.")

        # Apply super-resolution to the restored image
        print(f"[3/3] Applying Super-Resolution...")
        final_image = super_resolve(restored_image)
        print(f"      -> Super-resolution complete.")

        # Save the final image to a temporary file and return the path
        output_dir = Path("output/final_images")
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path_obj = Path(image_path)
        final_image_path = output_dir / f"{input_path_obj.stem}_restored{input_path_obj.suffix}"
        final_image.save(str(final_image_path))
        print(f"      -> Final image saved to: {final_image_path}")
        
        print("--- Pipeline Finished Successfully ---")
        return final_image_path
    except Exception as e:
        print(f"\nAn error occurred during the pipeline: {e}")
        raise


def main():
    p = argparse.ArgumentParser(description="Uruchomienie pipeline'u do ekstrakcji cech z obrazu.")
    p.add_argument("image", type=str, help="Ścieżka do obrazu wejściowego.")
    args = p.parse_args()

    hole_generator = ImageHoleGenerator(holes=1, points=4)

    try:
        features = process_image(args.image)
        cluster = predict_cluster(features.cpu())
        hole_image = hole_generator.apply_to_image(args.image)
        restored_image = apply_inpainting(hole_image, cluster_id=cluster)
        image_sr = super_resolve(restored_image)
        image_sr.show()
        print("\nPrzetwarzanie zakończone sukcesem.")
    except Exception as e:
        print(f"\nWystąpił błąd podczas przetwarzania: {e}")

if __name__ == "__main__":
    main()