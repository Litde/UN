from datasets import load_dataset
import cv2
from holes_generator import ImageHoleGenerator


def test_load_dataset():
    gen = ImageHoleGenerator(holes=3, points=3, debug=True)
    gen.load_image("image.jpg")

    corrupted, mask, rgba = gen.apply()


if __name__ == "__main__":
    test_load_dataset()