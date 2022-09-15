import os
import sys
sys.path.append(r"./CFG")
sys.path.append(r"./models")
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


def inference(model:dict[str:nn.Module], loader:DataLoader, monitor):
    pbar = tqdm(loader, desc="[inference]", file=sys.stdout)
    for _, (img, label) in enumerate(loader):
        img = img.to(cfg.device)


if __name__ == "__main__":
    if not os.path.isdir(cfg.sin_ckpt_path):
        os.makedirs(cfg.sin_ckpt_path)
    monitor = dynamic_pic(2000)
    encoder = UnetEncoder(latent_dim=cfg.latent_dim).to(cfg.device)
    fea_compress = feature_compress().to(cfg.device)

    ckpt = torch.load(cfg.resume_path)
    encoder.load_state_dict(ckpt["encoder"])
    fea_compress.load_state_dict(ckpt["feature_compress"])
    model = {"encoder":encoder, "fea_compress":fea_compress}

    monitor = dynamic_pic(2000)
    mapdict, train_loader = flowers_dataloader()
    print(mapdict)

    inference(model, train_loader, monitor)

