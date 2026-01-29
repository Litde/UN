from datasets import load_dataset
from tqdm import tqdm
import os
from PIL import Image
import io

# --- SETTINGS ---
OUTPUT_DIR = "wikiart"
SUPER_RESOLUTION_OUTPUT_DIR = "wikiart_super_res"

NUM_IMAGES = 5000
DATASET_PATH = "Artificio/WikiArt_Full"
SUPER_RESOLUTION_DATASET_PATH = "huggan/wikiart"

ADD_LABELS_TO_FILENAME = False
DOWNLOAD_SUPER_RESOLUTION = True

# --- CREATE FOLDER ---
output_directory = SUPER_RESOLUTION_OUTPUT_DIR if DOWNLOAD_SUPER_RESOLUTION else OUTPUT_DIR
os.makedirs(output_directory, exist_ok=True)

# --- LOAD DATASET (streaming = True saves memory) ---
if DOWNLOAD_SUPER_RESOLUTION:
    ds = load_dataset(SUPER_RESOLUTION_DATASET_PATH, split="train", streaming=True)
else:
    ds = load_dataset(DATASET_PATH, split="train", streaming=True)

print("Starting download...")

saved_count = 0

for sample in tqdm(ds):
    style = sample["style"]
    genre = sample["genre"]

    # Get image
    img_bytes = sample["image"]

    if isinstance(img_bytes, bytes):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        img = img_bytes

    if ADD_LABELS_TO_FILENAME:
        img.save(os.path.join(output_directory, f"{saved_count:04d}_{style:02d}_{genre:02d}.jpg"))
    else:
        img.save(os.path.join(output_directory, f"{saved_count:04d}.jpg"))

    saved_count += 1

    if saved_count % 20 == 0:
        print(f"Saved {saved_count} images...")

    # if saved_count >= NUM_IMAGES:
    #     break

print("Done! Images saved in:", output_directory)
