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
from models.encoders import Encoder, Decoder, feature_compress, Discriminator, UnetEncoder, U2Encoder

# set batch size to 1 when evaluate

def draw_hist(l1:list,label:str,weights, bins=10,  density=True, color='g', alpha=1, stacked=False):
    for i, lst in enumerate(l1):
        plt.hist(lst, bins=bins,range=[0,1], density=density, color=color[i], alpha=alpha, label=label[i], log=False, stacked=stacked, weights=weights[i] )
    
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
        if cfg.net_type == "U2Net":
            representation = model["fea_compress"](model["encoder"](img)[-1])
        else:
            representation = model["fea_compress"](model["encoder"](img))
        batch_scores = model["sin"](representation)
        for i in range(label.shape[0]):
            # pdb.set_trace()
            anomly_score[label[i].item()].append(batch_scores[i].item())
    
    color_lst = ["r", "g", "b", "y", "purple", "k", "c" ]
    # alpha_lst = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    for k, lst in enumerate(join_lst):
        plt.figure()
        tmp_lst = []
        keys_lst = []
        clr_lst = []
        weights = []
        for i, key in enumerate(lst):
            tmp_lst.append(anomly_score[mapdict[key]])
            keys_lst.append(key)
            clr_lst.append(color_lst[i])
            weights.append(np.ones_like(anomly_score[mapdict[key]])/len(anomly_score[mapdict[key]]))
        draw_hist(tmp_lst, keys_lst, color=clr_lst, alpha=0.5, bins=100, stacked=False, density=True, weights=weights)
        # plt.xlabel("score")
        # plt.ylabel("distribution (%)")
        plt.title("score distribution")
        plt.legend()
        plt.savefig("v2_score_distribution_mi_" + "_".join(lst))
        
        # plt.show()


if __name__ == "__main__":
    if not os.path.isdir(cfg.sin_ckpt_path):
        os.makedirs(cfg.sin_ckpt_path)
    monitor = dynamic_pic(2000)
    if cfg.net_type == "U2Net":
        encoder = U2Encoder(latent_dim=cfg.latent_dim).to(cfg.device)
    else:
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

    # join_lst = [["a_lg1000", "lg10000", "0175", "0039"], ["0039", "0175", "a_lg1000"], ["0039", "0104", "a_lg1000"], ["0037", "0038", "0039", "a_lg1000"]]
    join_lst = [["a_lg1000", "lg10000"], ["a_lg1000", "0037", "0038"], ["a_lg1000", "0038", "0039"], ["a_lg1000", "0104", "0175"], ["a_lg1000", "0104", "0039"]]
    inference(model, train_loader, mapdict, join_lst)
