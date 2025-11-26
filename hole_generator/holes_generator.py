from typing import Any, Generator

import numpy as np
import cv2
import random

from numpy import ndarray
from tqdm import tqdm


from tqdm import tqdm
import cv2
import numpy as np
import random
import os

class ImageHoleGenerator:
    def __init__(self, holes:int=1, points:int=5, debug:bool=False) -> None:
        self.debug = debug
        self.holes = holes
        self.points = points
        self.image = None
        self.output_image = None
        self.num_of_iteration = 0

    def load_image(self, image_pth:str) -> None:
        self.image = cv2.imread(image_pth)[:, :, ::-1]

    def _random_polygon(self, h, w):
        """Generate one jagged, irregular polygon like scribbles."""
        assert h > 0 and w > 0 or self.image is None, "Image must be loaded or valid dimensions provided."

        cx = random.randint(int(0.1*w), int(0.9*w))
        cy = random.randint(int(0.1*h), int(0.9*h))
        max_radius = min(h, w) // 6
        radius = random.randint(max_radius // 4, max_radius)

        points = []
        angle = 0
        while angle < 2 * np.pi:
            angle_step = random.uniform(np.pi/12, np.pi/4)
            r = radius * random.uniform(0.3, 1.0)
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            points.append([x, y])
            angle += angle_step

        return np.array(points, dtype=np.int32)

    def generate_holes(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate holes and return corrupted image and mask."""
        h, w, _ = self.image.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        for _ in range(self.holes):
            poly = self._random_polygon(h, w)
            cv2.fillPoly(mask, [poly], 1)

        corrupted = self.image.copy()
        corrupted[mask == 1] = 0

        if self.debug:
            cv2.imshow("Holes", corrupted[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return corrupted, mask

    def _save_all(self, corrupted, mask, output):
        os.makedirs("../output/images", exist_ok=True)
        # os.makedirs("../output/masks", exist_ok=True)
        # os.makedirs("../output/outputs", exist_ok=True)

        cv2.imwrite(f"../output/images/corrupted_{self.num_of_iteration}.png", corrupted[:, :, ::-1])
        # cv2.imwrite(f"../output/masks/mask{self.num_of_iteration}.png", mask * 255)
        # cv2.imwrite(f"../output/outputs/output{self.num_of_iteration}.png", output[:, :, ::-1])

    def apply(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.image is None:
            raise ValueError("No image loaded. Call load_image first.")

        corrupted, mask = self.generate_holes()
        mask_channel = mask[..., None]

        output = np.concatenate([corrupted, mask_channel], axis=2)

        if self.debug:
            print("Corrupted shape:", corrupted.shape)
            print("Mask shape:", mask_channel.shape)
            print("Output shape:", output.shape)

        self._save_all(corrupted, mask_channel, output)

        return corrupted, mask_channel, output

    def iterate_images(self, image_paths:list[str]) -> None:
        progress = tqdm(total=len(image_paths), desc="Processing images")
        for image_pth in image_paths:
            self.load_image(image_pth)
            self.apply()
            self.num_of_iteration += 1
            progress.update(1)
        progress.close()






