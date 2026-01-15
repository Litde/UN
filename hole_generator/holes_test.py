from holes_generator import ImageHoleGenerator
import os
from tqdm import tqdm

def test_load_dataset():
    image_paths = []
    pth = "../wikiart"
    progress = tqdm(total=len(os.listdir(pth)), desc="Loading image paths")
    for file_name in os.listdir(pth):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(pth, file_name))
        progress.update(1)
    progress.close()

    gen = ImageHoleGenerator(holes=1, points=4, debug=False)
    gen.iterate_images(image_paths)


def for_clusters_test():
    root_pth = "..\\UN\\clusters"

    parent_dir = os.path.dirname(root_pth)
    corrupted_root = os.path.join(parent_dir, "corrupted_clusters")
    os.makedirs(corrupted_root, exist_ok=True)

    for cluster_name in os.listdir(root_pth):
        cluster_pth = os.path.join(root_pth, cluster_name)
        if not os.path.isdir(cluster_pth):
            continue

        corrupted_cluster_pth = os.path.join(corrupted_root, cluster_name)

        image_paths = []
        files = os.listdir(cluster_pth)
        progress = tqdm(files, desc=f"Loading image paths for {cluster_name}")

        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(cluster_pth, file_name))
        progress.close()

        gen = ImageHoleGenerator(
            holes=1,
            points=4,
            debug=False,
            output_dir=corrupted_cluster_pth,
            recreate_output_dir=False
        )

        gen.iterate_images(image_paths)



if __name__ == "__main__":
    for_clusters_test()