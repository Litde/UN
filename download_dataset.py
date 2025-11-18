from datasets import load_dataset
import os
from PIL import Image
import io

# --- SETTINGS ---
OUTPUT_DIR = "./wikiart_200"
NUM_IMAGES = 200

# --- CREATE FOLDER ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD DATASET (streaming = True saves memory) ---
ds = load_dataset("huggan/wikiart", split="train", streaming=True)

print("Starting download...")

# --- DOWNLOAD FIRST 200 ---
for i, sample in enumerate(ds):
    if i >= NUM_IMAGES:
        break

    img_bytes = sample["image"]  # PIL Image already decoded

    # If image is raw bytes:
    if isinstance(img_bytes, bytes):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        img = img_bytes

    img.save(os.path.join(OUTPUT_DIR, f"{i:04d}.jpg"))

    if i % 20 == 0:
        print(f"Saved {i} images...")

print("Done! Images saved in:", OUTPUT_DIR)

