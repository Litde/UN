from unet_autoencoder import UNet
from utils import plot_unet_architecture
import netron
import torch


def main():
    model = UNet()
    plot_unet_architecture(model, input_shape=(1, 3, 256, 256), filename="architecture/unet_architecture")

if __name__ == '__main__':
    main()