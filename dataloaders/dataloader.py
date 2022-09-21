import os
import sys
import pdb
import torch
sys.path.append(r"/media/gauthierli-org/GauLi1/code/生仝智能/representationAE/CFG/cfg.py")

import cfg as CFG
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
from typing import Dict
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

# 1 dataloader
class cell_seg_dataset(Dataset):
    def __init__(self, root:str=CFG.data_path, transform=None, file_list=None, train=True) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.train = train
        if file_list is None:
            self.imgs = self._get_imgs_name(list(os.listdir(root)))
        else:
            self.imgs = file_list

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image = self.imgs[index]
        img = os.path.join(self.root,image + ".png")
        img = Image.open(img).convert("RGB")
        label = os.path.join(self.root, image + "_label.png")
        # label = Image.open(label).convert("L")
        if self.transform is not None:
            img = self.transform(img)
            # img, label = self.transform(img), self.transform(label)
        return img, img# img, label

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

def cell_dataloader(nw=0):
    transform = T.Compose([T.RandomCrop((CFG.img_size, CFG.img_size)),T.ToTensor()])
    dataset = cell_seg_dataset(transform=transform)
    return DataLoader(dataset=dataset,batch_size=CFG.bs,num_workers=nw,shuffle=True,pin_memory=True,collate_fn=default_collate, drop_last=True)

def flowers_dataloader(root=CFG.data_path,transform=None,num_class=CFG.num_class):
    transform = T.Compose([T.RandomCrop((CFG.img_size, CFG.img_size)),T.ToTensor()])
    ds = ImageFolder(root,transform=transform, target_transform=T.Lambda(lambda y:torch.eye(num_class, dtype=torch.float32)[y]))
    return ds.class_to_idx, DataLoader(ds, batch_size=CFG.bs,num_workers=CFG.nw,shuffle=True,pin_memory=True, collate_fn=default_collate, drop_last=True)


if __name__ == "__main__":
    map_dict, loader = flowers_dataloader()
    print(map_dict)
    print(len(loader))
    for img, label in loader:
        print(label)
        print(img.shape, label)
        break