import os
import sys
import pdb
import torch
sys.path.append(r"F:\representationAE(version2)\CFG")

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
    def __init__(self, root:str, transform=None, file_list=None, train=True) -> None:
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

def cell_dataloader(dataset, nw=0):
    return DataLoader(dataset=dataset,batch_size=CFG.bs,num_workers=nw,shuffle=True,pin_memory=True,collate_fn=default_collate)

def flowers_dataloader(transform=None):
    transform = T.Compose([T.Resize(CFG.img_size),T.ToTensor(),T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    ds = ImageFolder(CFG.data_path,transform=transform, target_transform=T.Lambda(lambda y:torch.eye(5)[y]))
    return ds.class_to_idx, DataLoader(ds, batch_size=CFG.bs,num_workers=CFG.nw,shuffle=True,pin_memory=True, collate_fn=default_collate)


if __name__ == "__main__":
    map_dict, loader = flowers_dataloader()
    print(map_dict)
    print(len(loader))
    for img, label in loader:
        print(label)
        print(img.shape, label)
        break