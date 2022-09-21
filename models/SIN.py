import os
import sys
sys.path.append(r"/media/gauthierli-org/GauLi1/code/生仝智能/representationAE/CFG")
sys.path.append(r"/media/gauthierli-org/GauLi1/code/生仝智能/representationAE/models")
import pdb
import torch
import torch.nn as nn
import CFG.cfg as cfg
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from utils.vis import dynamic_pic
from models.miNets import TotalMI
from torch.utils.data import Dataset, DataLoader
from dataloaders.dataloader import flowers_dataloader
from models.encoders import Encoder, Decoder, feature_compress, Discriminator, UnetEncoder

class SIN(nn.Module):
    def __init__(self, latent_dim=cfg.latent_dim):
        super(SIN, self).__init__()
        if self.training:
            self.dropout = nn.Dropout1d(0.5)
        else:
            self.dropout = nn.Sequential(*[])
        self.lin = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(),
                                 nn.Linear(64, 128), nn.ReLU(), self.dropout,
                                 nn.Linear(128, 256), nn.ReLU(), self.dropout, 
                                #  nn.Linear(256, 512), nn.ReLU(), nn.Dropout1d(0.5), 
                                #  nn.Linear(512, 256), nn.ReLU(),nn.Dropout1d(0.5),
                                 nn.Linear(256,128), nn.ReLU(),self.dropout, 
                                 nn.Linear(128, 64), nn.ReLU(),self.dropout,
                                 nn.Linear(64, 1))

    def forward(self, x):
        return nn.Sigmoid()(self.lin(x))

