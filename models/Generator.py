import sys
import torch
import torch.nn as nn

from models.base_model import *


class GeneratorResNet(nn.Module):
    def __init__(self):
        super(GeneratorResNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1),
                                     #nn.BatchNorm2d(128),
                                     nn.PReLU(128),
                                   )

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.stn_1 = Net_32_old()

        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1),
                                   nn.GroupNorm(num_groups=8, num_channels=64),
                                   nn.PReLU(64)
                                   )

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.stn_2 = Net_64_old()
        self.CBAM = CBAM(64)

        # Residual blocks
        res_blocks = []
        for _ in range(2):
            res_blocks.append(ResidualBlock_64(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.stn_3 = Net_64_old()

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv5 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),
                                   nn.GroupNorm(num_groups=8, num_channels=32),
                                   nn.PReLU(32)
                                   )

        # Residual blocks
        res_blocks_32 = []
        for _ in range(2):
            res_blocks_32.append(ResidualBlock_32(32))
        self.res_blocks2 = nn.Sequential(*res_blocks_32)

        self.conv6 = nn.Sequential(nn.Conv2d(32, 12, 5, 1, 2),
                                   nn.GroupNorm(num_groups=3, num_channels=12),
                                   nn.PReLU(12)
                                   )

        self.conv7 = nn.Sequential(nn.Conv2d(12, 3, 3, 1, 1),
                                   nn.Tanh())


    def forward(self, x):
        x = self.conv1(x)
        x = self.up1(x)
        #print(x.shape)
        x, heatmap_1 = self.stn_1(x)
        x = self.conv2(x)
        x = self.up2(x)
        a = x
        for _ in range(3):
            x_stn, heatmap_2 = self.stn_2(a)
            x_stn_att = self.CBAM(x_stn)
            #x_all = self.res_blocks(x_all)
            x_res = self.res_blocks(x_stn_att)
            a = x_stn + x_res
        x_aligned, heatmap_3 = self.stn_3(a)
        x_end = self.up3(x_aligned)
        x_end = self.conv5(x_end)
        #x_end, heatmap_2 = self.stn_3(x_end)
        x_end = self.res_blocks2(x_end)
        x_end = self.conv6(x_end)
        x_end = self.conv7(x_end)
        return x_end, heatmap_1, heatmap_2, heatmap_3


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                   m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)



class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                   m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)