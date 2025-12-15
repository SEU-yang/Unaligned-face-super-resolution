import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import numpy as np
from torchvision.models import vgg19


# Define a residual block
class Residual(nn.Module):
    def __init__(self, ins, outs):
        super(Residual, self).__init__()
        self.convBlock = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins, 34, 1),
            nn.BatchNorm2d(34),
            nn.ReLU(inplace=True),
            nn.Conv2d(34, 34, 3, 1, 1),
            nn.BatchNorm2d(34),
            nn.ReLU(inplace=True),
            nn.Conv2d(34, outs, 1)
        )
        if ins != outs:
            self.skipConv = nn.Conv2d(ins, outs, 1)
        self.ins = ins
        self.outs = outs

    def forward(self, x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        return x


# Define a hourglass block
class HourGlassBlock(nn.Module):
    def __init__(self, dim, n, norm_layer):
        super(HourGlassBlock, self).__init__()
        self._dim = dim  ##128
        self._n = n  ##3
        self._norm_layer = norm_layer
        self._init_layers(self._dim, self._n, self._norm_layer)

    def _init_layers(self, dim, n, norm_layer):
        setattr(self, 'res' + str(n) + '_1', Residual(dim, dim))
        setattr(self, 'pool' + str(n) + '_1', nn.MaxPool2d(2, 2))
        setattr(self, 'res' + str(n) + '_2', Residual(dim, dim))
        if n > 1:
            self._init_layers(dim, n - 1, norm_layer)
        else:
            self.res_center = Residual(dim, dim)
        setattr(self, 'res' + str(n) + '_3', Residual(dim, dim))
        setattr(self, 'unsample' + str(n), nn.Upsample(scale_factor=2))

    def _forward(self, x, dim, n):
        up1 = x
        up1 = eval('self.res' + str(n) + '_1')(up1)
        low1 = eval('self.pool' + str(n) + '_1')(x)
        low1 = eval('self.res' + str(n) + '_2')(low1)
        if n > 1:
            low2 = self._forward(low1, dim, n - 1)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.' + 'res' + str(n) + '_3')(low3)
        up2 = eval('self.' + 'unsample' + str(n)).forward(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._forward(x, self._dim, self._n)


## define spatial transform network
class Net_32(nn.Module):
    def __init__(self):
        super(Net_32, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.MaxPool2d(2, stride=2),
            nn.PReLU(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.MaxPool2d(2, stride=2),
            nn.PReLU(32),
            nn.Conv2d(32, 20, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=5, num_channels=20),
            nn.MaxPool2d(2, stride=2),
            nn.PReLU(20)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(20 * 4 * 4, 20),
            nn.PReLU(20),
            nn.Linear(20, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 20 * 4 * 4)
        theta = torch.tanh(self.fc_loc(xs))
        theta = theta.view(-1, 2, 3)
        print(theta[0])
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x


## define spatial transform network
class Net_32_old(nn.Module):
    def __init__(self):
        super(Net_32_old, self).__init__()
        self.face = nn.Sequential(nn.Conv2d(128, 68, 3, 1, 1),
                                  nn.GroupNorm(num_groups=17, num_channels=68),
                                  nn.PReLU(68))
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.fan = HourGlassBlock(68, 3, nn.BatchNorm2d)

        self.down1 = nn.MaxPool2d(2, stride=2)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(196, 96, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(num_groups=12, num_channels=96),
            nn.MaxPool2d(2, stride=2),
            nn.PReLU(96),
            nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.MaxPool2d(2, stride=2),
            nn.PReLU(32),
            nn.Conv2d(32, 20, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=5, num_channels=20),
            nn.MaxPool2d(2, stride=2),
            nn.PReLU(20)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(20 * 4 * 4, 20),
            nn.PReLU(20),
            nn.Linear(20, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        x_trans = self.face(x)
        x_trans = self.up1(x_trans)
        heatmap_64 = self.fan(x_trans)
        heatmap_32 = self.down1(heatmap_64)
        # print("heatmap_shape:", heatmap.shape)
        x_all = torch.cat([x, heatmap_32], 1)
        xs = self.localization(x_all)
        xs = xs.view(-1, 20 * 4 * 4)
        theta = torch.tanh(self.fc_loc(xs))
        theta = theta.view(-1, 2, 3)
        print(theta[0])
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x, heatmap_64

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x


class Net_64_old(nn.Module):
    def __init__(self):
        super(Net_64_old, self).__init__()
        # Spatial transformer localization-network
        self.face = nn.Sequential(nn.Conv2d(64, 68, 3, 1, 1),
                                  nn.GroupNorm(num_groups=17, num_channels=68),
                                  nn.PReLU(68))

        self.fan = HourGlassBlock(68, 3, nn.BatchNorm2d)

        self.localization = nn.Sequential(
            nn.Conv2d(132, 96, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(num_groups=12, num_channels=96),
            nn.PReLU(96),
            nn.Conv2d(96, 64, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.MaxPool2d(2, stride=2),
            nn.PReLU(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.MaxPool2d(2, stride=2),
            nn.PReLU(32),
            nn.Conv2d(32, 20, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=5, num_channels=20),
            nn.MaxPool2d(2, stride=2),
            nn.PReLU(20),
            nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=5, num_channels=20),
            nn.MaxPool2d(2, stride=2),
            nn.PReLU(20)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(20 * 4 * 4, 20),
            nn.PReLU(20),
            nn.Linear(20, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        x_trans = self.face(x)
        heatmap = self.fan(x_trans)
        # print("heatmap_shape:", heatmap.shape)
        x_all = torch.cat([x, heatmap], 1)
        xs = self.localization(x_all)
        xs = xs.view(-1, 20 * 4 * 4)
        theta = torch.tanh(self.fc_loc(xs))
        # theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        print(theta[0])
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x, heatmap

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x


# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # [B, C, 1, 1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.PReLU(in_planes // ratio),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in [3, 7], 'kernel_size must be 3 or 7'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        return self.sigmoid(self.conv(x_cat))


# Full CBAM Module
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, ratio=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ResidualBlock_32(nn.Module):
    def __init__(self, num_channels, num_groups=8):
        super(ResidualBlock_32, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=num_channels),
            nn.PReLU(num_channels),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        )

    def forward(self, x):
        return x + self.block(x)  # 残差连接


class ResidualBlock_64(nn.Module):
    def __init__(self, num_channels, num_groups=16):
        super(ResidualBlock_64, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=num_channels),
            nn.PReLU(num_channels),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        )

    def forward(self, x):
        return x + self.block(x)  # 残差连接


class Net_128(nn.Module):
    def __init__(self):
        super(Net_128, self).__init__()
        # Spatial transformer localization-network
        # Spatial transformer localization-network
        self.face = nn.Sequential(nn.Conv2d(32, 68, 3, 1, 1),
                                  nn.BatchNorm2d(68),
                                  nn.ReLU())

        self.down = nn.MaxPool2d(2, stride=2)
        self.fan = HourGlassBlock(68, 3, nn.BatchNorm2d)

        self.localization = nn.Sequential(
            nn.Conv2d(136, 96, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(96, 72, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(72),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(72, 48, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(48, 20, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(20 * 4 * 4, 20),
            nn.ReLU(True),
            nn.Linear(20, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        x_trans = self.face(x)
        x_trans = self.down(x_trans)
        heatmap = self.fan(x_trans)
        # print("heatmap_shape:", heatmap.shape)
        x_all = torch.cat([x_trans, heatmap], 1)
        xs = self.localization(x_all)
        xs = xs.view(-1, 20 * 4 * 4)
        theta = torch.tanh(self.fc_loc(xs))
        # theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        # print(theta[0])
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x, heatmap

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x
