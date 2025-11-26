import cv2

from unet_autoencoder import UNet
from utils import plot_unet_architecture
import netron
import torch


def main():
    model = UNet()
    img = cv2.imread('wikiart/0000.jpg', cv2.IMREAD_COLOR)
    print(img.shape)

if __name__ == '__main__':
    main()