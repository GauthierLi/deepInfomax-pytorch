import os
import pdb
import torch

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
import matplotlib.pyplot as plt 

import sys
sys.path.append("./utils")
sys.path.append("./model")

from vis import *
from PIL import Image
from tqdm import tqdm
from compressAndScore import *
from model.compressAndScore import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    bs = 2
    lr = 1e-5
    wd = 1e-4
    epoch = 150
    img_size = (512, 512)
    representation_dim = 2

    data_path = r"/media/gauthierli-org/GauLi1/data/cell_segmentation/train_data"
    save_path = r"./checkpoint"

    show_training_pic = True

    resume = False
    resume_path = r"/media/gauthierli-org/GauLi1code/生仝智能/representationAE/checkpoint/best.pth"

# utils
def make_folders(fold_name):
    if not os.path.exists(fold_name):
        os.makedirs(fold_name)

# 1 dataloader
class cell_seg_dataset(Dataset):
    def __init__(self, root:str, transform=None, file_list=None, train=True) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.train = train
        if file_list is None:
            self.imgs = self._get_imgs_name(list(os.listdir(root)))[:6000]
        else:
            self.imgs = file_list

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image = self.imgs[index]
        img = os.path.join(self.root,image + ".png")
        img = Image.open(img).convert("RGB")
        label = os.path.join(self.root, image + "_label.png")
        label = Image.open(label).convert("L")
        if self.transform is not None:
            img, label = self.transform(img), self.transform(label)
        return img, label

    @staticmethod
    def _get_imgs_name(raw_list:list) -> list:
        clean_list = []
        for img in raw_list:
            img_name = img.split(".")[0]
            if img_name.split("_")[-1] != "label":
                clean_list.append(img_name)
        return clean_list

    @staticmethod
    def collate_fn(batch):
        pass

def build_dataloader(dataset, nw=0):
    return DataLoader(dataset=dataset,batch_size=CFG.bs,num_workers=nw,shuffle=True,pin_memory=True,collate_fn=default_collate)

# 2 trandform
def build_transform():
    return T.Compose([T.Resize(CFG.img_size), T.ToTensor()])

# 3 train one epoch
def train_one_epoch(nce:nn.Module, discriminator:nn.Module, dataloader:DataLoader, MI_optimizer, D_optimizer):
    nce.train()
    discriminator.train()
    nce.to(CFG.device)
    # discriminator.to(CFG.device)
    # show_rep = dynamic_pic("representation", style=".")
    show_loss = dynamic_pic("loss", style="")

    mi_loss = 0
    train_count = 0
    N = float(len(dataloader))
    pbar = tqdm(dataloader, total=N, file=sys.stdout)
    for num, (imgs, _) in enumerate(pbar):
        assert imgs.shape[-1] == nce.inp_sh, "Img shape doesnot match model needs ..."
        # ---------------------------------------------------------------
        # 1 first train discriminator
        # ---------------------------------------------------------------
        if train_count < 200:
            MI_optimizer.zero_grad()
            D_optimizer.zero_grad()
            representation = nce.encoder(imgs.to(CFG.device))
            representation = nce.compressNet(representation)
            sample_data = get_gaussian_sampler(CFG.bs, CFG.representation_dim)
            
            discriminate_loss = discriminator(representation) - discriminator(sample_data.to(CFG.device)) + 1
            discriminate_loss.backward()

            if CFG.show_training_pic:
                show_loss(num, discriminate_loss.item())

            train_count += 1
            D_optimizer.step()
        
        if train_count >= 200 and train_count < 400:
            # ---------------------------------------------------------------
            # 2 encoder fool discriminator
            # ---------------------------------------------------------------
            MI_optimizer.zero_grad()
            D_optimizer.zero_grad()

            representation = nce.encoder(imgs.to(CFG.device))
            representation = nce.compressNet(representation)
            # if CFG.show_training_pic:
            #     x, y = representation[0].cpu().detach().numpy()
            #     show_rep(x, y)

            fool_discriminator_loss = 1 - discriminator(representation)
            fool_discriminator_loss.backward()

            MI_optimizer.step()
            train_count += 1
        
        
            # ---------------------------------------------------------------
            # 3 third train encoder
            # ---------------------------------------------------------------
            MI_optimizer.zero_grad()
            D_optimizer.zero_grad()

            MI_loss = nce(imgs.to(CFG.device))
            MI_loss.backward()

            MI_optimizer.step()


            pbar.desc = f"[training] loss: {MI_loss.item()}"
            mi_loss += MI_loss.item()

        if train_count >= 400:
            train_count = 0

    print("loss: {:.5f} lr: {:.6f}".format(2 * mi_loss/N, MI_optimizer.param_groups[0]['lr']), flush=True)
    return mi_loss/N
    
# 4 model & loss & optimizer & lr scheduler & save checkpoint
if __name__ == "__main__":
    make_folders(CFG.save_path)
    encoder = Encoder(3,CFG.representation_dim)
    compressNetwork = compressNet((16,16), dropout=0)
    if CFG.resume:
        assert os.path.isfile(CFG.resume_path), "checkpoints not exists ... ..."
        ckpt = torch.load(CFG.resume_path)
        encoder.load_state_dict(ckpt["encoder"])
        compressNetwork.load_state_dict(ckpt["compressNetwork"])

    local_network = LocalScore(CFG.representation_dim)
    global_network = GlobalScore(CFG.representation_dim)

    nce = InfoNCELossNet(encoder=encoder, compressNet=compressNetwork, local_var_network=local_network, global_var_network=global_network, temperature=1.).to(CFG.device)
    discriminator = Discriminator(CFG.representation_dim).to(CFG.device)

    pa = get_parameter_number(nce)["Total"]
    print("paramter number: {}".format(pa))

    transform = build_transform()
    train_dataset = cell_seg_dataset(CFG.data_path, transform=transform)
    train_loader = build_dataloader(dataset=train_dataset)
    
    MI_optimizer = torch.optim.AdamW(nce.parameters(),lr=CFG.lr, weight_decay=CFG.wd)
    D_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    MI_lr_schedule = torch.optim.lr_scheduler.ExponentialLR(MI_optimizer, gamma=0.98)
    D_lr_schedule = torch.optim.lr_scheduler.ExponentialLR(D_optimizer, gamma=0.98)

    best_loss = 999
    for epoch in range(CFG.epoch):

        loss = train_one_epoch(nce=nce, discriminator=discriminator, dataloader=train_loader, MI_optimizer=MI_optimizer, D_optimizer=D_optimizer)
        MI_lr_schedule.step()
        D_lr_schedule.step()
        if loss < best_loss:
            best = best_loss
            torch.save({"encoder": encoder.state_dict(), "compressNetwork": compressNetwork.state_dict()}, os.path.join(CFG.save_path, "best.pth"))
