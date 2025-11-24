from sympy.physics.quantum.circuitplot import render_label

from basic_inpainting import BasicInpainting, import_params
from unet_autoencoder import UNet, prepare_input
import torch
from torchview import draw_graph
import os
import cv2


def main():
    model = UNet()

    out = model(prepare_input("wikiart_200/0000.jpg", True))
    print(f'Output shape: {out.shape}')



if __name__ == '__main__':
    main()