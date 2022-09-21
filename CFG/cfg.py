import os
import torch

# general parameters
nw = 0
bs = 8
lr = 5e-3
wd = 1e-3
epoch = 1500
img_size = 128
latent_dim = 128
num_class = 2
net_type = "U2Net"
device = "cuda" if torch.cuda.is_available else "cpu"

# data path
log_img = "./log_img"
data_path = r"F:\data\lg1" 
ckpt_path = r"./checkpoint"
sin_ckpt_path = r"./sincheckpoint"
sin_log_img = "./sin_log_img"

# resume
resume = False
resume_path = r"F:\representationAE(version2)\c1\best_epoch.pth"

# SIN train
sin_train_encoder_ckpt = r"F:\representationAE(version2)\lg_inference_use\encoder_ckpt\epoch_1499.pth"

# inference
inference_data = r"F:\data\all_data"
inference_use_encoder_ckpt = sin_train_encoder_ckpt
inference_use_sin_ckpt = r"F:\representationAE(version2)\lg_inference_use\sin_ckpt\sin_epoch_332.pth"