import os
from dataset import ImageMaskDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm

from comet_ml import Experiment

class BasicInpainting(nn.Module):
    def __init__(self):
        super(BasicInpainting, self).__init__()
        self.encoder_module = None
        self.latent_dim = 512
        self.inpainting_module = None
        self.decoder_module = None
        self.super_res_module = None




if __name__ == '__main__':
    pass