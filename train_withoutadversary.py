import os
import sys
from tkinter import Variable
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
from dataloaders.dataloader import cell_dataloader, flowers_dataloader
from models.encoders import Encoder, Decoder, feature_compress, Discriminator, UnetEncoder, U2Encoder, U2Decoder

def train_one_epoch(epoch:int, models:dict, loader:DataLoader, monitor, mapdict=None):
    total_recons_loss, total_mi_loss = 0., 0.

    N = len(loader)
    pbar = tqdm(loader,desc=f"[epoch {epoch} training]", total=N,file=sys.stdout)

    recons_lf= nn.MSELoss()
    bce_lf = nn.BCEWithLogitsLoss()

    for num, (img, label) in enumerate(pbar):
        img = img.to(cfg.device)
        label = label.to(cfg.device)
        
        # if False:#(epoch + 1) % 3 != 0:
        # train encoder decoder
        models["encoder"]["net"].train()
        models["decoder"]["net"].train()
        models["feature_compress"]["net"].eval()
        models["mi"]["net"].eval()
        models["encoder"]["optim"].zero_grad()
        models["decoder"]["optim"].zero_grad()

        if cfg.net_type == "U2Net":
            features = models["encoder"]["net"](img)
            feature = features[-1]
            reconstruct = models["decoder"]["net"](features)
            # pdb.set_trace()
            recons_loss = 0
            for rec in reconstruct:
                recons_loss += torch.pow(1 + recons_lf(img * 25, rec * 25), 4)
        else:
            feature = models["encoder"]["net"](img)
            reconstruct = models["decoder"]["net"](feature)
            recons_loss = recons_lf(img * 25, reconstruct * 25) 
            

        total_recons_loss += recons_loss.item() 
        monitor(num, recons_loss.item(), category="recons", mode="line", drop_x=True)
        monitor((255 * img[0].permute(1,2,0).cpu().detach().numpy()).astype("uint8"), 0, category="ori", mode="figure")
        if cfg.net_type == "U2Net":
            monitor((255 * reconstruct[0][0].permute(1,2,0).cpu().detach().numpy()).astype("uint8"), 0, category="rec", mode="figure")
        else:
            monitor((255 * reconstruct[0].permute(1,2,0).cpu().detach().numpy()).astype("uint8"), 0, category="rec", mode="figure")

        
        recons_loss.backward()
        models["encoder"]["optim"].step()
        models["decoder"]["optim"].step()
        # else:
        # train mi
        models["encoder"]["net"].train()
        models["decoder"]["net"].train()
        models["feature_compress"]["net"].train()
        models["mi"]["net"].train()
        models["encoder"]["optim"].zero_grad()
        models["decoder"]["optim"].zero_grad()
        models["feature_compress"]["optim"].zero_grad()
        models["mi"]["optim"].zero_grad()
        
        if cfg.net_type == "U2Net":
            feature = models["encoder"]["net"](img)[-1]
        else:
            feature = models["encoder"]["net"](img)
        representation = models["feature_compress"]["net"](feature)
        
        mi_loss = models["mi"]["net"](feature, representation, label) + torch.norm(representation, p=2, dim=1).mean()
        total_mi_loss += mi_loss.item()
        monitor(num, mi_loss.item(), category="mi loss", drop_x=True)
        for i, rep in enumerate(representation):
            l = rep.cpu().detach().numpy().tolist()
            x ,y = l[0], l[1]
            cate = label[i].cpu().detach().numpy().argmax()
            if mapdict is not None:
                monitor(x,y,category=mapdict[cate], mode="scatter")
            else:
                monitor(x,y,category=f"rep{cate}", mode="scatter")

        
        mi_loss.backward()
        models["encoder"]["optim"].step()
        models["decoder"]["optim"].step()
        models["feature_compress"]["optim"].step()
        models["mi"]["optim"].step()

        # draw 
        if num % 1 == 0:
            monitor.draw(joint=["recons", "mi loss", "ori", "rec", [f"rep{i}" for i in range(cfg.num_class)]],
                        row_max=2, pause=0, save_path=os.path.join(cfg.log_img, "view.png"))
    
    total = total_recons_loss / N + total_mi_loss / N
    print("epoch:{}, total loss:{:.4f}, lr:{:8f}".format(epoch, total, models["mi"]["optim"].param_groups[0]['lr']))
    monitor.draw(joint=["recons", "mi loss", "ori", "rec", [f"rep{i}" for i in range(cfg.num_class)]], 
                    row_max=2, pause=0, save_path=os.path.join(cfg.log_img, f"epoch{epoch}.png"))
    monitor.clean([f"rep{i}" for i in range(cfg.num_class)])
    return total

if __name__ == "__main__":
    if not os.path.isdir(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)
    if not os.path.isdir(cfg.log_img):
        os.makedirs(cfg.log_img)
    
    monitor = dynamic_pic(2000)
    mapdict, train_loader = flowers_dataloader()
    print(mapdict)
    # {'ST16Rm-LG-MD-ML-PG-SD-032-2-0037': 0, 'ST16Rm-LG-MD-ML-PG-SD-032-2-0038': 1, 'ST16Rm-LG-MD-ML-PG-SD-032-4-0091': 2, 'ST16Rm-LG-MD-ML-PG-SD-032-4-0093': 3, 'a_lg1000': 4}

    if cfg.net_type == "U2Net":
        encoder = U2Encoder(latent_dim=cfg.latent_dim).to(cfg.device)
        decoder = U2Decoder().to(cfg.device)
        mi_loss = TotalMI().to(cfg.device)
    else:
        encoder = UnetEncoder(latent_dim=cfg.latent_dim).to(cfg.device)
        decoder = Decoder().to(cfg.device)
        mi_loss = TotalMI().to(cfg.device)
    fea_compress = feature_compress().to(cfg.device)
    discrim = Discriminator().to(cfg.device)

    optim_en = optim.AdamW(encoder.parameters(), lr=cfg.lr,weight_decay=cfg.wd)
    optim_de = optim.AdamW(decoder.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    optim_feac = optim.AdamW(fea_compress.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    optim_mi = optim.AdamW(mi_loss.parameters(), lr=cfg.lr)
    optim_discrim = optim.AdamW(discrim.parameters(), lr=cfg.lr, weight_decay=cfg.wd)


    lrscd_en = optim.lr_scheduler.ExponentialLR(optim_en, gamma=0.1)
    lrscd_de = optim.lr_scheduler.ExponentialLR(optim_de, gamma=0.1)
    lrscd_feac = optim.lr_scheduler.ExponentialLR(optim_feac, gamma=0.1)
    lrscd_mi = optim.lr_scheduler.ExponentialLR(optim_mi, gamma=0.1)
    lrscd_discrim = optim.lr_scheduler.ExponentialLR(optim_discrim, gamma=0.1)

    # lrscd_en = optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_en, T_0=10, T_mult=2, eta_min=1e-8)
    # lrscd_de = optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_de, T_0=10, T_mult=2, eta_min=1e-8)
    # lrscd_feac = optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_feac, T_0=10, T_mult=2, eta_min=1e-8)
    # lrscd_mi = optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_mi, T_0=10, T_mult=2, eta_min=1e-8)
    # lrscd_discrim = optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_discrim, T_0=10, T_mult=2, eta_min=1e-8)


    models = {"encoder": {"net":encoder, "optim":optim_en, "lr_scd": lrscd_en},
              "decoder": {"net":decoder, "optim":optim_de, "lr_scd": lrscd_de},
              "feature_compress": {"net":fea_compress, "optim":optim_feac, "lr_scd": lrscd_feac},
              "mi":{"net":mi_loss, "optim":optim_mi, "lr_scd": lrscd_mi},
              "discriminator":{"net":discrim, "optim":optim_discrim, "lr_scd": lrscd_discrim}}
    
    if cfg.resume:
        ckpt = torch.load(cfg.resume_path)
        for key in models:
            models[key]["net"].load_state_dict(ckpt[key])

    best_loss = 999999999999999
    for epoch in range(cfg.epoch):
        loss = train_one_epoch(epoch, models,train_loader, monitor)
        for key in models:
            models[key]["lr_scd"].step()

        ckpt = dict()
        for key in models:
            ckpt[key] = models[key]["net"].state_dict()
        torch.save(ckpt, os.path.join(cfg.ckpt_path, f"epoch_{epoch}.pth"))
        if loss < best_loss:
            best_loss = loss
            torch.save(ckpt, os.path.join(cfg.ckpt_path, f"best_epoch.pth"))