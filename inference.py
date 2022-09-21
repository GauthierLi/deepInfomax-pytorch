import os
import sys
sys.path.append(r"./CFG")
sys.path.append(r"./models")
import pdb
import torch
import numpy as np 
import torch.nn as nn
import CFG.cfg as cfg
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from models.SIN import SIN
from utils.vis import dynamic_pic
from models.miNets import TotalMI
from torch.utils.data import Dataset, DataLoader
from dataloaders.dataloader import flowers_dataloader
from models.encoders import Encoder, Decoder, feature_compress, Discriminator, UnetEncoder

# set batch size to 1 when evaluate

def draw_hist(l1:list,label:str, bins=20,  density=True, color='g', alpha=1):
    n, bins, patches = plt.hist(l1, bins=bins,  density=density, color=color, alpha=alpha, label=label, log=False)
    plt.legend()
    # plt.plot(bins[:-1],n,'--')

def inference(model:dict[str:nn.Module], loader:DataLoader, mapdict,  join_lst):
    inverse_mapdict = dict()
    for key in mapdict:
        inverse_mapdict[mapdict[key]] = key
    pbar = tqdm(loader, desc="[inference]", file=sys.stdout)
    anomly_score = dict()
    for key in mapdict:
        anomly_score[mapdict[key]] = []
    for _, (img, label) in enumerate(pbar):
        img = img.to(cfg.device)
        label = torch.argmax(label, axis=1)
        representation = model["fea_compress"](model["encoder"](img))
        batch_scores = model["sin"](representation)
        for i in range(label.shape[0]):
            anomly_score[label[i].item()].append(batch_scores[i].item())
    
    color_lst = ["r", "g", "b", "y", "purple", "k", "c" ]
    alpha_lst = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    for k, lst in enumerate(join_lst):
        plt.figure()
        for i, key in enumerate(lst):
            draw_hist(anomly_score[mapdict[key]], key, color=color_lst[i], alpha=alpha_lst[i], bins=20)
        plt.xlabel("score")
        plt.ylabel("distribution (%)")
        plt.title("score distribution")
        # plt.savefig("score_distribution_mi_" + "_".join(lst))
        
        plt.show()


if __name__ == "__main__":
    if not os.path.isdir(cfg.sin_ckpt_path):
        os.makedirs(cfg.sin_ckpt_path)
    monitor = dynamic_pic(2000)
    encoder = UnetEncoder(latent_dim=cfg.latent_dim).to(cfg.device)
    fea_compress = feature_compress().to(cfg.device)
    sin = SIN().to(cfg.device)
    encoder.eval()
    fea_compress.eval()
    sin.eval()

    rep_ckpt = torch.load(cfg.inference_use_encoder_ckpt)
    sin_ckpt = torch.load(cfg.inference_use_sin_ckpt)
    encoder.load_state_dict(rep_ckpt["encoder"])
    fea_compress.load_state_dict(rep_ckpt["feature_compress"])
    sin.load_state_dict(sin_ckpt)
    model = {"encoder":encoder, "fea_compress":fea_compress, "sin":sin}

    monitor = dynamic_pic(2000)
    mapdict, train_loader = flowers_dataloader(root=cfg.inference_data, num_class=7)
    print(mapdict)

    join_lst = [["a_lg1000", "lg10000", "0175", "0039"], ["0039", "0175", "a_lg1000"], ["0039", "0104", "a_lg1000"], ["0037", "0038", "0039", "a_lg1000"]]
    inference(model, train_loader, mapdict, join_lst)
