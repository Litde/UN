from datasets import load_dataset
from tqdm import tqdm
import os
from PIL import Image
import shutil
import io

# --- SETTINGS ---
OUTPUT_DIR = "wikiart"
NUM_IMAGES = 500

# --- CREATE FOLDER ---
shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD DATASET (streaming = True saves memory) ---
ds = load_dataset("Artificio/WikiArt_Full", split="train", streaming=True)

print("Starting download...")

saved_count = 0
for sample in tqdm(ds):
    # Check style
    if sample["genre"] != 'portrait':
        continue

    # Get image
    img_bytes = sample["image"]

    if isinstance(img_bytes, bytes):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        img = img_bytes

    img.save(os.path.join(OUTPUT_DIR, f"{saved_count:04d}.jpg"))
    saved_count += 1

    if saved_count % 20 == 0:
        print(f"Saved {saved_count} images...")

    if saved_count >= NUM_IMAGES:
        break

print("Done! Images saved in:", OUTPUT_DIR)
