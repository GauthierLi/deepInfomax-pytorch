import os
import sys
sys.path.append(r"/media/gauthierli-org/GauLi1/code/生仝智能/representationAE/CFG")
sys.path.append(r"/media/gauthierli-org/GauLi1/code/生仝智能/representationAE/models")
import pdb
import torch
import numpy as np
import CFG.cfg as cfg
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from models.SIN import SIN
from utils.vis import dynamic_pic
from models.miNets import TotalMI
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from dataloaders.dataloader import flowers_dataloader
from models.encoders import Encoder, Decoder, feature_compress, Discriminator, UnetEncoder

if __name__ == "__main__":
    if not os.path.isdir(cfg.sin_ckpt_path):
        os.makedirs(cfg.sin_ckpt_path)

    if not os.path.isdir(cfg.sin_log_img):
        os.makedirs(cfg.sin_log_img)
    monitor = dynamic_pic(2000)
    encoder = UnetEncoder(latent_dim=cfg.latent_dim).to(cfg.device)
    fea_compress = feature_compress().to(cfg.device)

    ckpt = torch.load(cfg.inference_use_encoder_ckpt)
    encoder.load_state_dict(ckpt["encoder"])
    fea_compress.load_state_dict(ckpt["feature_compress"])

    SINNet = SIN().to(cfg.device)

    ckpt = torch.load(cfg.inference_use_sin_ckpt)
    SINNet.load_state_dict(ckpt)
    opt = torch.optim.AdamW(SINNet.parameters(), lr=1e-3, weight_decay=cfg.wd)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-8)

    mapdict, dl = flowers_dataloader(root=r"F:\data\lg1")
    print(mapdict)
    best_loss = 99999999999999
    N = len(dl)
    for epoch in range(cfg.epoch):
        total_loss = 0.
        encoder.eval()
        fea_compress.eval()
        SINNet.train()

        SINNet.train()
        pbar = tqdm(dl, desc="[Train]", file=sys.stdout)
        for num, (img, label) in enumerate(pbar):
            feature = encoder(img.to(cfg.device))
            rep = fea_compress(feature)

            label = torch.argmax(label, dim=1).float().to(cfg.device)
            
            pred = SINNet(rep).squeeze()
            loss = torch.pow(nn.MSELoss()(pred , label) + 1, 11)
            monitor(num, loss.item(),category="loss", drop_x=True)
            for i in range(pred.shape[0]):
                monitor(pred[i].item(), label[i].item(), category=f"{int(label[i].item())}", mode="scatter")
            total_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()
            if num % 10 == 0:
                monitor.draw([["0", "1"], "loss"], row_max=2)

        monitor.clean([0, 1])


        lr_scheduler.step()
        total_loss /= N

        if total_loss < best_loss:
            best_loss = total_loss
            print("saving best ... ...")
            torch.save(SINNet.state_dict(), os.path.join(cfg.sin_ckpt_path, "sin_best_epoch.pth"))
        print("saving last ... ...")
        torch.save(SINNet.state_dict(), os.path.join(cfg.sin_ckpt_path, f"sin_epoch_{epoch}.pth"))
        plt.savefig(os.path.join(cfg.sin_log_img, "result.png"))


            

        
