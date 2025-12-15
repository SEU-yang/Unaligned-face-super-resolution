import torch
import torch.nn as nn
from torchvision.models import vgg19
from models.FAN import FAN
from torch.utils.model_zoo import load_url
import torch.nn.functional as F


class FAN_heatmap(nn.Module):
    def __init__(self):
        super(FAN_heatmap, self).__init__()
        FAN_net = FAN(4)
        FAN_model_url = 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar'
        fan_weights = load_url(FAN_model_url, map_location=lambda storage, loc: storage)
        FAN_net.load_state_dict(fan_weights)
        for p in FAN_net.parameters():
            p.requires_grad = False
        self.FAN_net = FAN_net        

    def forward(self, data):
        heat_gt = self.FAN_net(data) 
        #heatmap_gt = torch.cat(heat_gt,0)      
        return heat_gt[0]


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor1_2 = nn.Sequential(*list(vgg19_model.features.children())[:4])
        self.feature_extractor2_2 = nn.Sequential(*list(vgg19_model.features.children())[:9])
        self.feature_extractor3_4 = nn.Sequential(*list(vgg19_model.features.children())[:18])
        self.feature_extractor4_4 = nn.Sequential(*list(vgg19_model.features.children())[:27])
        self.feature_extractor5_4 = nn.Sequential(*list(vgg19_model.features.children())[:36])

    def forward(self, img):
        f1_2 = self.feature_extractor1_2(img)
        f2_2 = self.feature_extractor2_2(img)
        f3_4 = self.feature_extractor3_4(img)
        f4_4 = self.feature_extractor4_4(img)
        f5_4 = self.feature_extractor5_4(img)
        return f1_2, f2_2, f3_4, f4_4, f5_4


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)

