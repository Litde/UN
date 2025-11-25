from holes_generator import ImageHoleGenerator
import os
from tqdm import tqdm

def test_load_dataset():
    image_paths = []
    pth = "K:/Polibuda/Sezon_02_Semestr_02/UN/wikiart"
    progress = tqdm(total=len(os.listdir(pth)), desc="Loading image paths")
    for file_name in os.listdir(pth):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(pth, file_name))
        progress.update(1)
    progress.close()

    gen = ImageHoleGenerator(holes=1, points=4, debug=False)
    gen.iterate_images(image_paths)


if __name__ == "__main__":
    test_load_dataset()