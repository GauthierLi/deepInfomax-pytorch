from msilib.schema import Feature
import os
import sys
import pdb
import torch
sys.path.append(r"F:\representationAE(version2)\CFG")

import cfg

import numpy as np
import torch.nn as nn


class GlobalMI(nn.Module):
    """with great parameters of weight decays !!!!!!!!!"""
    def __init__(self, latent_dim=cfg.latent_dim):
        super(GlobalMI, self).__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(latent_dim*2, latent_dim,  kernel_size=1)
        self.bn1 = nn.BatchNorm2d(latent_dim)
        self.conv2 = nn.Conv2d(latent_dim, 1, kernel_size=1)
        self.activate = nn.LeakyReLU(inplace=True)

    def forward(self, features, representation):
        B,C,W,H = features.shape
        representation = torch.stack([representation for i in range(H*W)],dim=2).reshape(B,C,H,W)
        c = torch.cat([features, representation], dim=1)

        out = self.activate(self.bn1(self.conv1(c))) 
        out = nn.Sigmoid()(self.conv2(out).squeeze()) # 是否要加 sigmoid??????? 不加容易梯度爆炸，加了梯度消失
        return out.mean(dim=(1,2))


class LocalMI(nn.Module):
    def __init__(self):
        super(LocalMI, self).__init__()
    def forward(self, features, representation):
        B,C,W,H = features.shape
        representation = torch.stack([representation for i in range(H*W)],dim=2).reshape(B,C,H,W)

        features_norm = torch.stack([torch.norm(features, p=2, dim=1) for i in range(C)], dim=1)
        representation_norm = torch.stack([torch.norm(representation, p=2, dim=1) for i in range(C)], dim=1)

        features = torch.div(features, features_norm)
        representation = torch.div(representation, representation_norm)

        out = (features * representation).sum(dim=1) 
        return out.mean(dim=(1,2))

if __name__ == "__main__":
    tst_features = torch.ones(4, cfg.latent_dim, 8, 8)
    tst_rep = torch.randn(4, cfg.latent_dim)

    gmi = GlobalMI()
    lmi = LocalMI()
    print(gmi(tst_features, tst_rep))
    print(lmi(tst_features, tst_rep))