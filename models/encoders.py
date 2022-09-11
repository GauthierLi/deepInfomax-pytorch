import os
import sys
import pdb
import torch
from typing import Dict
sys.path.append(r"/media/gauthierli-org/GauLi1/code/生仝智能/representationAE/CFG")
import cfg

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import mobilenet_v3_small, resnet101, resnet50
from collections import OrderedDict

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        x = self.up(x1)
        x = self.conv(x)
        return x

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

class Encoder(nn.Module):
    r"""
        >>> "resnet": features channel 2048
        >>> "mobilenet": features channel 576
    """
    def __init__(self, mode="mobilenet", latent_dim=cfg.latent_dim):
        super(Encoder,self).__init__()
        self.mode = mode
        self.model_zoo = {"mobilenet":mobilenet_v3_small(), "resnet101":resnet101(), "resnet50":resnet50()}
        if mode == "mobilenet":
            self.features = self.model_zoo[mode].features
            self.out_ch = 576
        else:
            self.features = IntermediateLayerGetter(self.model_zoo[mode], 
            dict([(name, name) for name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']]))
            self.out_ch = 2048
        self.cbr = nn.Sequential(nn.Conv2d(self.out_ch, latent_dim, kernel_size=1), nn.BatchNorm2d(latent_dim), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        if self.mode == "mobilenet":
            return self.cbr(self.features(x))
        else:
            return self.cbr(self.features(x)["layer4"])

class Decoder(nn.Module):
    def __init__(self,mode, latent_dim=cfg.latent_dim, shape_template=[], img_size=cfg.img_size):
        super(Decoder, self).__init__()
        if mode == "mobilenet":
            shape_template = [2, 576, img_size // 32, img_size // 32]
        elif mode == "resnet":
            shape_template = [2, 2048, img_size // 32, img_size // 32]
        _, _, _, self.size = shape_template
        self.channel = latent_dim
        up_times = (np.log2(img_size) - np.log2(self.size)).astype("uint8")
        self.up = nn.Sequential(*[Up(self.channel , self.channel) for i in range(up_times)])
        
        self.out = nn.Conv2d(self.channel, 3, kernel_size=1)
    
    def forward(self, x):
        x = self.up(x)
        
        x = self.out(x)
        return x

class feature_compress(nn.Module):
    def __init__(self, img_size=cfg.img_size):
        super(feature_compress, self).__init__()

        self.linear1 = nn.Linear((img_size // 32)**2, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, 1)
        self.activate = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):
        B, C, _,_ = x.shape
        x = x.view(B, C, -1)

        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return x.squeeze(dim=2)

class Discriminator(nn.Module):
    """2 class classification, use BCEloss"""
    def __init__(self, latent_dim=cfg.latent_dim):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 256)
        self.linear2 = nn.Linear(256, 2)

        self.activate = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.linear2(x)
        return nn.Softmax(dim=1)(x)


if __name__ == "__main__":
    tst = torch.ones((2,3,256,256))
    e1 = Encoder("mobilenet")
    d1 = Decoder("mobilenet")
    fea_comp = feature_compress()

    discrim = Discriminator()
    
    feature = e1(tst)
    print("feature shape:", feature.shape)
    print("reconstract shape check:", d1(feature).shape)
    print("representation shape check:", fea_comp(feature).shape)
    print("discrim check:", discrim(fea_comp(feature)).shape, discrim(fea_comp(feature))[0])