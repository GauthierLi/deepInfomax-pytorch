import torch
# general parameters
bs = 2
nw = 0
lr=1e-4
wd = 1e-3
epoch = 500
img_size = 256
latent_dim = 512
device = "cuda" if torch.cuda.is_available else "cpu"

# data path
data_path = r"/home/gauthierli-org/data/flower_photos"
ckpt_path = r".checkpoint/"
log_img = "./log_img"
