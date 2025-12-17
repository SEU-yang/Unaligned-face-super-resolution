import torch
import numpy as np
import time
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from tensorboardX import SummaryWriter
import argparse
from models.Generator import *
from PIL import Image


torch.manual_seed(1)    # reproducible
os.makedirs("testing_lfw_results/all", exist_ok=True)
os.makedirs("testing_lfw_results/sr", exist_ok=True)
os.makedirs("testing_lfw_results/gt", exist_ok=True)
os.makedirs("testing_lfw_results/lr", exist_ok=True)
os.makedirs("testing_lfw_results/lro", exist_ok=True)



parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--image_dir", type=str, default="../../Data/Helen", help="The path of the training dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr_G", type=float, default=1e-3, help="adam: learning rate")
parser.add_argument("--lr_D", type=float, default=1e-3, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--imsize", type=int, default=128, help="image size")
parser.add_argument("--hr_height", type=int, default=128, help="hr_height")
parser.add_argument("--hr_width", type=int, default=128, help="hr_width")
parser.add_argument("--channel", type=int, default=3, help="channel")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between model checkpoints")
parser.add_argument("--save_path", type=str, default="loss_recoder")
parser.add_argument("--d_out_shape", type=int, default=1)
opt = parser.parse_args()


cuda = True if torch.cuda.is_available() else False


# Define Dataloader
lr_transforms = [transforms.Resize((opt.imsize // 8, opt.imsize // 8)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                 ]

hr_transforms = [transforms.Resize((opt.imsize, opt.imsize)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                 ]

hr_transforms_256 = [transforms.Resize((opt.imsize * 2, opt.imsize * 2)),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                  ]


class custom_dataset(Dataset):
    def __init__(self, txt_path, lr_transforms=None, hr_transforms=None, hr_transforms_256=None):
        self.transform_lr = transforms.Compose(lr_transforms)  # 传入数据预处理
        self.transform_gt = transforms.Compose(hr_transforms)
        self.hr_transform_256 = transforms.Compose(hr_transforms_256)
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        self.img_list_0 = [i.split()[0] for i in lines]  # 得到所有的unaligned image names
        self.img_list_1 = [i.split()[1] for i in lines]  # 得到所有的aligned image names

    def __getitem__(self, idx):  # 根据 idx 取出其中一个
        img_0 = Image.open(self.img_list_0[idx % len(self.img_list_0)]).convert('RGB')
        img_1 = Image.open(self.img_list_1[idx % len(self.img_list_1)]).convert('RGB')
        img_lr = self.transform_lr(img_0)
        img_hr = self.transform_gt(img_1)
        img_gt256 = self.hr_transform_256(img_1)
        return {'lr': img_lr, 'hrgt': img_hr, 'hrgt_256': img_gt256}

    def __len__(self):  # 总数据的多少
        return len(self.img_list_0)


testing_loader = DataLoader(custom_dataset('lfw_testing_list.txt', lr_transforms=lr_transforms, hr_transforms=hr_transforms, hr_transforms_256=hr_transforms_256), batch_size=opt.batch_size, shuffle=False,
                    num_workers=1)


#### Initialize SRGAN Network
generator = GeneratorResNet()

uplr_near = nn.Upsample(scale_factor=8, mode='nearest')
uplr_bili = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)


Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

if cuda:
    generator = generator.cuda()


# Load model training weights
generator.load_state_dict(torch.load("saved_models/generator_60.pth"))


for i, imgs in enumerate(testing_loader):
            print("Current batch : {}".format(i))
        
            imgs_lr = imgs['lr'].cuda()
            imgs_hr = imgs['hrgt'].cuda()

            imgs_lr_up1 = uplr_near(imgs_lr)
            imgs_lr_up2 = uplr_bili(imgs_lr)

            gen_hr, heatmap_1, heatmap_2, heatmap_3 = generator(imgs_lr)


            batches_done = i
            
            if batches_done % 1 == 0:
               save_image(torch.cat((imgs_lr_up1.data, gen_hr.data, imgs_hr.data), -2),
                           'testing_lfw_results/all/%03d.png' % batches_done, normalize=True)
               save_image(imgs_lr.data,'testing_lfw_results/lro/%03d_lr.png' % batches_done, normalize=True)
               save_image(imgs_lr_up1.data,'testing_lfw_results/lr/%03d.png' % batches_done, normalize=True)
               #save_image(imgs_lr_up2.data,'testing_surveillance_results/%03d_lr2.png' % batches_done, normalize=True)
               save_image(gen_hr.data,'testing_lfw_results/sr/%03d.png' % batches_done, normalize=True)
               save_image(imgs_hr.data,'testing_lfw_results/gt/%03d.png' % batches_done, normalize=True)

