import os
import sys
sys.path.append(r"/media/gauthierli-org/GauLi1/code/生仝智能/representationAE/CFG")
sys.path.append(r"/media/gauthierli-org/GauLi1/code/生仝智能/representationAE/models")
import pdb
import torch
import torch.nn as nn
import CFG.cfg as cfg
import torch.optim as optim

from tqdm import tqdm
from utils.vis import dynamic_pic
from models.miNets import TotalMI
from torch.utils.data import Dataset, DataLoader
from dataloaders.dataloader import flowers_dataloader
from models.encoders import Encoder, Decoder, feature_compress, Discriminator

def train_one_epoch(epoch:int, models:dict, loader:DataLoader, monitor):
    total_recons_loss, total_mi_loss = 0., 0.

    N = len(loader)
    pbar = tqdm(loader,desc=f"[epoch {epoch} training]", total=N,file=sys.stdout)

    recons_lf= nn.MSELoss()
    bce_lf = nn.BCEWithLogitsLoss()

    for num, (img, label) in enumerate(pbar):
        img = img.to(cfg.device)
        label = label.to(cfg.device)
        
        # train encoder decoder
        models["encoder"]["net"].train()
        models["decoder"]["net"].train()
        models["feature_compress"]["net"].eval()
        models["mi"]["net"].eval()
        models["encoder"]["optim"].zero_grad()
        models["decoder"]["optim"].zero_grad()


        feature = models["encoder"]["net"](img)
        reconstruct = models["decoder"]["net"](feature)
        recons_loss = recons_lf(img, reconstruct)

        total_recons_loss += recons_loss.item() 
        monitor(num, recons_loss.item(), category="recons", mode="line", drop_x=True)
        monitor((255 * img[0].permute(1,2,0).cpu().detach().numpy()).astype("uint8"), 0, category="ori", mode="figure")
        monitor((255 * reconstruct[0].permute(1,2,0).cpu().detach().numpy()).astype("uint8"), 0, category="rec", mode="figure")

        
        recons_loss.backward()
        models["encoder"]["optim"].step()
        models["decoder"]["optim"].step()

        # train mi
        models["encoder"]["net"].eval()
        models["decoder"]["net"].eval()
        models["feature_compress"]["net"].train()
        models["mi"]["net"].train()
        models["feature_compress"]["optim"].zero_grad()
        models["mi"]["optim"].zero_grad()

        feature = models["encoder"]["net"](img)
        representation = models["feature_compress"]["net"](feature)
        mi_loss = models["mi"]["net"](feature, representation)
        total_mi_loss += mi_loss.item()
        monitor(num, mi_loss.item(), category="mi loss", drop_x=True)
        for i, rep in enumerate(representation):
            l = rep.cpu().detach().numpy().tolist()
            x ,y = l[0], l[1]
            cate = label[i].cpu().detach().numpy().argmax()
            monitor(x,y,category=f"rep{cate}", mode="scatter")

        
        mi_loss.backward()
        models["feature_compress"]["optim"].step()
        models["mi"]["optim"].step()

        # draw 
        if num % 10 == 0:
            monitor.draw(joint=["recons", "mi loss", "ori", "rec", ["rep0", "rep1","rep2","rep3","rep4"]],
                        row_max=2, pause=0, save_path=os.path.join(cfg.log_img, "view.png"))
    
    total = total_recons_loss / N + total_mi_loss / N
    print("epoch:{}, total loss:{:.4f}, lr:{:8f}".format(epoch, total, models["mi"]["optim"].param_groups[0]['lr']))
    monitor.draw(joint=["recons", "mi loss", "ori", "rec", ["rep0", "rep1","rep2","rep3","rep4"]], 
                    row_max=2, pause=0, save_path=os.path.join(cfg.log_img, f"epoch{epoch}.png"))
    monitor.clean(["rep0", "rep1","rep2","rep3","rep4"])
    return total

if __name__ == "__main__":
    if not os.path.isdir(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)
    if not os.path.isdir(cfg.log_img):
        os.makedirs(cfg.log_img)
    
    monitor = dynamic_pic(2000)
    map_dic, train_loader = flowers_dataloader()

    encoder = Encoder("mobilenet").to(cfg.device)
    decoder = Decoder("mobilenet").to(cfg.device)
    fea_compress = feature_compress().to(cfg.device)
    mi_loss = TotalMI().to(cfg.device)
    discrim = Discriminator().to(cfg.device)

    optim_en = optim.AdamW(encoder.parameters(), lr=cfg.lr,weight_decay=cfg.wd)
    optim_de = optim.AdamW(decoder.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    optim_feac = optim.AdamW(fea_compress.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    optim_mi = optim.AdamW(mi_loss.parameters(), lr=cfg.lr)
    optim_discrim = optim.AdamW(discrim.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    lrscd_en = optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_en, T_0=10, T_mult=2, eta_min=1e-8)
    lrscd_de = optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_de, T_0=10, T_mult=2, eta_min=1e-8)
    lrscd_feac = optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_feac, T_0=10, T_mult=2, eta_min=1e-8)
    lrscd_mi = optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_mi, T_0=10, T_mult=2, eta_min=1e-8)
    lrscd_discrim = optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_discrim, T_0=10, T_mult=2, eta_min=1e-8)


    models = {"encoder": {"net":encoder, "optim":optim_en, "lr_scd": lrscd_en},
              "decoder": {"net":decoder, "optim":optim_de, "lr_scd": lrscd_de},
              "feature_compress": {"net":fea_compress, "optim":optim_feac, "lr_scd": lrscd_feac},
              "mi":{"net":mi_loss, "optim":optim_mi, "lr_scd": lrscd_mi},
              "discriminator":{"net":discrim, "optim":optim_discrim, "lr_scd": lrscd_discrim}}
    
    # for key in models:
    #     print(next(models[key]["net"].parameters()).device)
    best_loss = 999
    for epoch in range(cfg.epoch):
        loss = train_one_epoch(epoch, models,train_loader, monitor)
        for key in models:
            models[key]["lr_scd"].step()

        ckpt = dict()
        for key in models:
            ckpt[key] = models[key]["net"].state_dict()
        torch.save(ckpt, os.path.join(cfg.ckpt_path, f"epoch_{epoch}.png"))
        if loss < best_loss:
            best_loss = loss
            torch.save(ckpt, os.path.join(cfg.ckpt_path, f"best_epoch.png"))