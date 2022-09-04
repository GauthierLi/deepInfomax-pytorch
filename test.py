import sys
import torch
sys.path.append("./utils")
sys.path.append("./model")

from vis import dynamic_pic
from utils import dynamic_pic
from tqdm import tqdm

import matplotlib.pyplot as plt

dp = dynamic_pic("norm in torch", style=".", color="r")
for i in tqdm(range(10000), desc="norm", file=sys.stdout):
    x, y = (torch.randn(2) / 5.).numpy().tolist()
    dp(x, y)
plt.show()
