from holes_generator import ImageHoleGenerator
import os

def test_load_dataset():
    image_paths = []
    pth = "K:/Polibuda/Sezon_02_Semestr_02/UN/inputs"
    for img in os.listdir(pth):
        image_paths.append(os.path.join(pth, img))

    gen = ImageHoleGenerator(holes=3, points=3, debug=False)
    gen.iterate_images(image_paths)


if __name__ == "__main__":
    test_load_dataset()