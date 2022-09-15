import os
import sys
import pdb
import torch
sys.path.append(r"F:\representationAE(version2)\CFG")
sys.path.append(r"F:\representationAE(version2)\models")

import cfg

import numpy as np
import torch.nn as nn
from encoders import DoubleConv
from torch.autograd import Variable

# class GlobalMI(nn.Module):
#     """with great parameters of weight decays !!!!!!!!!"""
#     def __init__(self, latent_dim=cfg.latent_dim):
#         super(GlobalMI, self).__init__()
#         self.latent_dim = latent_dim
#         self.conv1 = DoubleConv(latent_dim*2, latent_dim, latent_dim)# nn.Conv2d(latent_dim*2, latent_dim,  kernel_size=1)
#         self.conv2 = nn.Conv2d(latent_dim, 1, kernel_size=1)
#         self.activate = nn.LeakyReLU(inplace=True)

#     def forward(self, features, representation):
#         B,C,W,H = features.shape
#         pdb.set_trace()
#         representation = torch.stack([representation for i in range(H*W)],dim=2).reshape(B,C,H,W)
#         c = torch.cat([features, representation], dim=1)

#         out = self.conv1(c)
#         out = nn.Sigmoid()(self.conv2(out).squeeze(dim=1)) # 是否要加 sigmoid??????? 不加容易梯度爆炸，加了梯度消失
#         return out.mean(dim=(1,2))

class GlobalMI(nn.Module):
    """with great parameters of weight decays !!!!!!!!!"""
    def __init__(self, latent_dim=cfg.latent_dim):
        super(GlobalMI, self).__init__()
        self.lin = nn.Sequential(nn.Linear(512+128, 1024, bias=True), nn.LeakyReLU(),
                                 nn.Linear(1024, 512, bias=True), nn.LeakyReLU(),
                                 nn.Linear(512,  256, bias=True), nn.LeakyReLU(),
                                 nn.Linear(256, 1, bias=True))
    
    def forward(self, features, representation):
        B,C,W,H = features.shape
        features = features.view(B, -1)
        x = torch.cat([features, representation],dim=1 )
        # pdb.set_trace()
        return nn.Sigmoid()(self.lin(x)).squeeze(dim=1)

# class LocalMI(nn.Module):
#     """with great parameters of weight decays !!!!!!!!!"""
#     def __init__(self, latent_dim=cfg.latent_dim):
#         super(LocalMI, self).__init__()
#         self.latent_dim = latent_dim
#         self.conv1 = nn.Conv2d(latent_dim*2, latent_dim,  kernel_size=1, bias=True)
#         self.conv2 = nn.Conv2d(latent_dim, 1, kernel_size=1, bias=True)
#         self.activate = nn.LeakyReLU(inplace=True)

#     def forward(self, features, representation):
#         B,C,W,H = features.shape
#         representation = torch.stack([representation for i in range(H*W)],dim=2).reshape(B,C,H,W)
#         c = torch.cat([features, representation], dim=1)

#         out = self.conv1(c)
#         out = nn.Sigmoid()(self.conv2(out).squeeze(dim=1)) # 是否要加 sigmoid??????? 不加容易梯度爆炸，加了梯度消失
#         return out.mean(dim=(1,2))

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


class TotalMI(nn.Module):
    def __init__(self):
        super().__init__()
        self.lmi = LocalMI()
        self.gmi = GlobalMI()
    
    def _get_focus(self, labels):
        B, _ = labels.shape
        mapmat = Variable(torch.zeros(B,B), requires_grad=False)
        for i in range(B):
            for j in range(B):
                if labels[i].equal(labels[j]):
                    mapmat[i][j] = 1
        return mapmat

    def forward(self, features, representation, labels):
        B,C,H,W = features.shape
        lmi_scoremap = torch.zeros(B,B)
        gmi_scoremap = torch.zeros(B,B)

        for i in range(B):
            for j in range(B):
                lmi_scoremap[i][j] = self.lmi(features[i].unsqueeze(dim=0), representation[j].unsqueeze(dim=0))
                gmi_scoremap[i][j] = self.gmi(features[i].unsqueeze(dim=0), representation[j].unsqueeze(dim=0))

        lmi_scoremap = nn.Softmax(dim=1)(lmi_scoremap)
        gmi_scoremap = nn.Softmax(dim=1)(gmi_scoremap)

        focus = Variable(torch.eye(B, dtype=torch.float32), requires_grad=False)# self._get_focus(labels)
        zero_focus = Variable(1 - focus , requires_grad=False)

        lmi_score = (zero_focus * lmi_scoremap).sum() - (focus * lmi_scoremap).sum()
        gmi_score = (zero_focus * gmi_scoremap).sum() - (focus * lmi_scoremap).sum()

        return (lmi_score + gmi_score)
    

if __name__ == "__main__":
    tst_features = torch.ones(4, cfg.latent_dim, 8, 8)
    tst_rep = torch.randn(4, cfg.latent_dim)

    gmi = GlobalMI()
    lmi = LocalMI()
    glo = TotalMI()
    print(gmi(tst_features, tst_rep))
    print(lmi(tst_features, tst_rep))
    # print(glo(tst_features, tst_rep, ))