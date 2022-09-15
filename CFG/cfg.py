import torch
# general parameters
bs = 16
nw = 0
lr=1e-3
wd = 1e-3
epoch = 1500
img_size = 64
latent_dim = 128
num_class = 2
device = "cuda" if torch.cuda.is_available else "cpu"

# data path
data_path = r"F:\data\lg1"
ckpt_path = r"./checkpoint"
log_img = "./log_img"
sin_ckpt_path = r"./sincheckpoint"

# resume
resume = False
resume_path = r"F:\representationAE(version2)\checkpoint\epoch_10.pth"