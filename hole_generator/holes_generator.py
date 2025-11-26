import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm


class ImageHoleGenerator:
    def __init__(self, holes: int = 1, points: int = 5, debug: bool = False) -> None:
        self.debug = debug
        self.holes = holes
        self.points = points
        self.image = None
        self.num = 0

        # pre-create output directory
        self.out_dir = Path("../output/images")
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # fast random generator
        self.rng = np.random.default_rng()

    def load_image(self, image_pth: str) -> None:
        img = cv2.imread(image_pth, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {image_pth}")
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _random_polygon(self, h: int, w: int) -> np.ndarray:
        """Generate a fast irregular polygon."""
        cx = self.rng.integers(int(0.1 * w), int(0.9 * w))
        cy = self.rng.integers(int(0.1 * h), int(0.9 * h))
        max_radius = min(h, w) // 6
        radius = self.rng.integers(max_radius // 4, max_radius)

        # generate angles + random radii in one vectorized block
        step_count = self.rng.integers(8, 20)
        angles = np.cumsum(self.rng.uniform(np.pi/12, np.pi/4, step_count))
        angles = angles[angles < 2 * np.pi]
        r = radius * self.rng.uniform(0.3, 1.0, len(angles))

        x = (cx + r * np.cos(angles)).astype(np.int32)
        y = (cy + r * np.sin(angles)).astype(np.int32)

        return np.stack([x, y], axis=1)

    def generate_holes(self) -> tuple[np.ndarray, np.ndarray]:
        h, w, _ = self.image.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # generate polygons
        polys = [self._random_polygon(h, w) for _ in range(self.holes)]
        cv2.fillPoly(mask, polys, 1)

        # apply holes
        corrupted = self.image.copy()
        corrupted[mask == 1] = 0

        return corrupted, mask

    def _save(self, corrupted: np.ndarray) -> None:
        # faster imwrite params
        path = self.out_dir / f"corrupted_{self.num}.png"
        cv2.imwrite(str(path), cv2.cvtColor(corrupted, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])

    def apply(self):
        corrupted, mask = self.generate_holes()
        mask_channel = mask[..., None]
        output = np.concatenate([corrupted, mask_channel], axis=2)

        if self.debug:
            print("Corrupted:", corrupted.shape)
            print("Mask:", mask_channel.shape)
            print("Output:", output.shape)

        self._save(corrupted)
        return corrupted, mask_channel, output

    def iterate_images(self, image_paths: list[str]) -> None:
        for p in tqdm(image_paths, desc="Processing"):
            self.load_image(p)
            self.apply()
            self.num += 1
