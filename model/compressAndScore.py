# ============================================================================ #
# Author:GauthierLi                                                            #
# Date: 2022/09/01                                                             #
# Email: lwklxh@163.com                                                        #
# ============================================================================ # 
import pdb
import torch

import torch.nn as nn

from u2net import U2NET
from torch.autograd import Variable
from torchvision.models import vgg16



class Encoder(nn.Module):
    def __init__(self,in_ch=3,out_ch=2):
        super(Encoder, self).__init__()
        self.encoder = U2NET(in_ch=in_ch,out_ch=in_ch)
        self.conv = nn.Conv2d(512, out_ch, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activate = nn.LeakyReLU(inplace=True)

    def forward(self,x):
        hx = x

        #stage 1
        hx1 = self.encoder.stage1(hx)
        hx = self.encoder.pool12(hx1)

        #stage 2
        hx2 = self.encoder.stage2(hx)
        hx = self.encoder.pool23(hx2)

        #stage 3
        hx3 = self.encoder.stage3(hx)
        hx = self.encoder.pool34(hx3)

        #stage 4
        hx4 = self.encoder.stage4(hx)
        hx = self.encoder.pool45(hx4)

        #stage 5
        hx5 = self.encoder.stage5(hx)
        hx = self.encoder.pool56(hx5)

        #stage 6
        hx6 = self.encoder.stage6(hx)
        out = self.bn(self.conv(hx6))

        return out # nn.Sigmoid()(out) - 0.5


class compressNet(nn.Module):
    """input (B,C,H,W) -> (BC,HW) -> (BC, 1) -> (B, C, 1)"""
    def __init__(self, feature_size:tuple, dropout=0.5):
        super(compressNet,self).__init__()
        self.linear1 = nn.Linear(feature_size[0] * feature_size[1], 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 128, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.linear4 = nn.Linear(128,64, bias=False)
        self.bn4 = nn.BatchNorm1d(64)
        self.linear5 = nn.Linear(64 ,1 ,bias=False)

        self.activate = nn.LeakyReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, feature):
        B,C,H,W = feature.shape
        feature_view = feature.view(B,C,-1)

        B,C,L = feature_view.shape
        feature_view = feature_view.reshape(B*C, L)

        out = self.drop(self.activate(self.bn1(self.linear1(feature_view))))
        out = self.drop(self.activate(self.bn2(self.linear2(out))))
        out = self.drop(self.activate(self.bn3(self.linear3(out))))
        out = self.drop(self.activate(self.bn4(self.linear4(out))))
        out = self.linear5(out)
        out = out.reshape(B, C, -1)

        return out # nn.Sigmoid()(out) - 0.5


class LocalScore(nn.Module):
    r"""B must be 1"""
    def __init__(self, feature_dim):
        super(LocalScore, self).__init__()
        self.feature_dim = feature_dim

    def forward(self, feature:torch.Tensor, high_level_feature:torch.Tensor):
        return self._batch_product(feature, high_level_feature)

    def _batch_product(self, feature:torch.Tensor, high_level_feature:torch.Tensor):
        result = []
        B, C, H, W = feature.shape
        assert (C == self.feature_dim and high_level_feature.shape[1] == self.feature_dim), "Channel Dim Error: feature dimension not matching ..."
        for i in range(B):
            result.append(feature[i].permute(1, 2, 0).reshape(-1, self.feature_dim) @ high_level_feature[i])
        result = torch.stack(result).mean(dim=1)
        return result


class GlobalScore(nn.Module):
    r"""
    low level feture B,C(feature_dim),H,W, high_level_feature B,C(feature_dim),L, B must be 1
    .. math::
    return score in $\mathbb{R}$
    """
    def __init__(self, feature_dim):
        super(GlobalScore, self).__init__()
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(self.feature_dim*2, 512, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)

        self.activate = nn.LeakyReLU(inplace=True)

    def forward(self, feature:torch.Tensor, high_level_feature:torch.Tensor):
        B ,C ,H ,W = feature.shape

        expanded_hlf = high_level_feature.repeat(1, 1, H*W).reshape(B ,C ,H ,W)
        combined_feature = torch.cat((feature, expanded_hlf), dim=1)

        out = self.activate(self.bn1(self.conv1(combined_feature)))
        out = self.activate(self.bn2(self.conv2(out)))
        out = self.activate(self.bn3(self.conv3(out)))
        out = self.activate(self.bn4(self.conv4(out)))
        out = nn.Sigmoid()(self.conv5(out))

        out = out.mean(dim=(2,3))
        return out


class Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super(Discriminator, self).__init__()
        self.feature_dim = feature_dim
        self.linear1 = nn.Linear(feature_dim, 8, bias=False)
        self.bn1 = nn.BatchNorm1d(8)
        self.linear2 = nn.Linear(8,1)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, representations):
        assert representations.shape[1] == self.feature_dim, "CHANNEL DIMENSION DOESN'T MATCH ... ..."
        out = self.activation(self.bn1(self.linear1(representations.squeeze(dim=2))))
        out = self.linear2(out)
        return nn.Sigmoid()(out).mean()


def get_gaussian_sampler(train_batch, z_dim):
    return Variable((torch.randn(train_batch, z_dim, dtype=torch.float32).unsqueeze(dim=2) + 2) / 5.)


class InfoNCELossNet(nn.Module):
    r"""
    need convert to device
    """
    inp_sh = 512
    def __init__(self ,encoder ,compressNet, global_var_network, local_var_network, temperature:float=1.):
        super(InfoNCELossNet, self).__init__()
        self.encoder = encoder
        self.compressNet = compressNet
        self.global_var_fn = global_var_network
        self.local_var_fn = local_var_network
        self.temperature = temperature

    def forward(self, imgs):
        features = self.encoder(imgs)
        representations = self.compressNet(features)
        g_score_map, l_score_map = [], []
        B, _, _, _ = features.shape
        for i in range(B):
            g_tmp_row = []
            l_tmp_row = []
            for j in range(B):
                g_tmp_row.append(self.global_var_fn(features[i].unsqueeze(dim=0), representations[j].unsqueeze(dim=0)).squeeze() / self.temperature)
                l_tmp_row.append(self.local_var_fn(features[i].unsqueeze(dim=0), representations[j].unsqueeze(dim=0)).squeeze() / self.temperature)
            g_score_map.append(torch.stack(g_tmp_row))
            l_score_map.append(torch.stack(l_tmp_row))
        l_score_map, g_score_map = torch.stack(l_score_map), torch.stack(g_score_map)
        l_score_map, g_score_map = nn.Softmax()(l_score_map), nn.Softmax()(g_score_map)

        mask = torch.eye(l_score_map.shape[0]).to(features.device)
        l_score_map, g_score_map = torch.log((l_score_map * mask).sum(dim=0)), torch.log((g_score_map * mask).sum(dim=0))
        MI_loss = -(l_score_map.sum() + g_score_map.sum())
        return MI_loss


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == "__main__":
    device = "cuda"
    ten1 = torch.ones((2,3,512,512)).to(device)
    encoder = Encoder(3,2).to(device)
    low_level_feature = encoder(ten1)
    print("low level:", low_level_feature.shape)

    # comp = compressNet((16,16)).to(device)
    # high_level_feature = comp(low_level_feature)
    # print("high level:", high_level_feature.shape)

    # Global_score = GlobalScore(2).to(device)
    # gscore = Global_score(low_level_feature, high_level_feature)
    # print("global score:", gscore.shape)

    # Local_score = LocalScore(2).to(device)
    # lscore = Local_score(low_level_feature, high_level_feature)
    # print("local score:", lscore.shape)

    # discriminator = AdversaryDiscriminator(2).to(device)
    # dscore = discriminator(high_level_feature)
    # print("discriminator score:", dscore.shape)

    # InfoNCE = InfoNCELossNet(encoder, GlobalScore(2), (16, 16)).to(device)
    # loss = InfoNCE(ten1)
    # print("loss:", loss)

    print(get_gaussian_sampler(3, 2))
