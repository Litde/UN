import numpy as np
import cv2
import random

class ImageHoleGenerator:
    def __init__(self, holes:int=1, points:int=5, debug:bool=False) -> None:
        self.debug = debug
        self.holes = holes
        self.points = points
        self.image = None
        self.output_image = None

    def load_image(self, image_pth:str) -> None:
        self.image = cv2.imread(image_pth)[:, :, ::-1]

    def _random_polygon(self, h, w):
        """Generate one irregular polygon inside the image."""
        assert h > 0 and w > 0 or self.image is None, "Image must be loaded or valid dimensions provided."

        # Random polygon center
        cx = random.randint(int(0.1*w), int(0.9*w))
        cy = random.randint(int(0.1*h), int(0.9*h))

        # Radius (size of the hole)
        max_radius = min(h, w) // 6
        radius = random.randint(max_radius // 3, max_radius)

        points = []
        for _ in range(self.points):
            angle = random.uniform(0, 2*np.pi)
            r = radius * random.uniform(0.5, 1.2)  # irregular radius
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            points.append([x, y])

        points = np.array(points, dtype=np.int32)
        return points

    def _save_all(self, corrupted, mask, output):
        """Save all images for debugging."""
        cv2.imwrite("output/corrupted.png", corrupted[:, :, ::-1])  # RGB to BGR
        cv2.imwrite("output/mask.png", mask * 255)  # mask is 0/1
        cv2.imwrite("output/output.png", output[:, :, ::-1])  # RGB to BGR

    def _save(self):
        """Save the output image with mask channel for debugging."""
        if self.output_image is not None:
            cv2.imwrite("output/output_with_mask.png", self.output_image[:, :, ::-1])  # RGB to BGR

    def apply(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = self.image

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")

        img = image.copy()
        h, w = img.shape[:2]

        # mask: (H, W)
        mask = np.zeros((h, w), dtype=np.uint8)

        # generate holes
        for _ in range(self.holes):
            poly = self._random_polygon(h, w)
            cv2.fillPoly(mask, [poly], 1)

        # corrupted: (H, W, 3)
        corrupted = img.copy()
        corrupted[mask == 1] = (0, 0, 0)

        # mask channel: (H, W, 1)
        mask_channel = mask[..., None]

        # output: (H, W, 4)
        try:
            output = np.concatenate([corrupted, mask_channel], axis=2)
        except Exception as e:
            raise RuntimeError(
                f"Failed to concatenate corrupted image {corrupted.shape} "
                f"and mask channel {mask_channel.shape}"
            ) from e

        if self.debug:
            print("Image shape:", img.shape)
            print("Corrupted image shape:", corrupted.shape)
            print("Mask shape:", mask_channel.shape)
            print("Output shape:", output.shape)
            self._save_all(corrupted, mask_channel, output)
        else:
            self._save()

        return corrupted, mask_channel, output





