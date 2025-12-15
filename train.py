import torch
import numpy as np
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import kornia
from tensorboardX import SummaryWriter
import argparse
from models.Generator import *
from models.function import *
from models.resnet import *
from PIL import Image
from torch.nn import DataParallel


torch.manual_seed(1)    # reproducible
os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--image_dir", type=str, default="../../Data/Helen", help="The path of the training dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--arcface_model", type=str, default="arcface/resnet18_110.pth", help="The path of the arcface model")
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
log_path = opt.save_path
writer = SummaryWriter(log_dir=log_path)
shape = (opt.hr_height, opt.hr_width)


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


train_loader = DataLoader(custom_dataset('training_list_total.txt', lr_transforms=lr_transforms, hr_transforms=hr_transforms,
                                   hr_transforms_256=hr_transforms_256), batch_size=opt.batch_size, shuffle=True,
                    num_workers=1)



arcface_model = resnet_face18(False)
arcface_model = DataParallel(arcface_model)
arcface_model.load_state_dict(torch.load(opt.arcface_model))


class ArcFaceIdentityLoss(nn.Module):
    def __init__(self):
        super(ArcFaceIdentityLoss, self).__init__()

        # Extracts features at the 11th layer
        self.feature_extractor = arcface_model


    def forward(self, img, gt):
        # b, c, h, w = img.shape
        # channel = 128
        images_vis_ycrcb = RGB2YCrCb(img)
        gt_vis_ycrcb = RGB2YCrCb(gt)

        sr_feat = images_vis_ycrcb[:, :1]
        hr_feat = gt_vis_ycrcb[:, :1]

        # 单位归一化（ArcFace通常会自己做）
        sr_feat = F.normalize(sr_feat, p=2, dim=1)
        hr_feat = F.normalize(hr_feat, p=2, dim=1)
        # cosine相似度 -> 损失
        cosine_sim = torch.sum(sr_feat * hr_feat, dim=1)  # [B]
        loss = 1 - cosine_sim  # [B]
        return loss.mean()



#### Initialize TDNN Network
generator = GeneratorResNet()
uplr = nn.Upsample(scale_factor=8, mode='bilinear')
uplr_2 = nn.Upsample(scale_factor=2, mode='bilinear')
discriminator = Discriminator(input_shape=(opt.channel, *shape))
FAN_heatmap = FAN_heatmap()
#color_loss = lab_color_loss()
arcface_id_loss = ArcFaceIdentityLoss()


feature_extractor = FeatureExtractor()
# Set feature extractor to inference mode
feature_extractor.eval()


# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    FAN_heatmap = FAN_heatmap.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()
    arcface_id_loss = arcface_id_loss.cuda()


generator._initialize_weights()
discriminator._initialize_weights()


# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(),lr = opt.lr_G)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(),lr = opt.lr_D)


# Define Tensor
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor


def adjust_learning_rate(epoch, lrr):
    ##Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lrr * (0.99 ** (epoch // 1))
    return lr


for epoch in range(opt.epoch, opt.n_epochs):
    print("Current epoch : {}".format(epoch))

    for i, imgs in enumerate(train_loader):
        print("Current batch : {}".format(i))

        # Configure model input
        imgs_lr = imgs['lr'].cuda()
        imgs_hr = imgs['hrgt'].cuda()
        imgs_hr256 = imgs['hrgt_256'].cuda()

        input_lrup = uplr(imgs_lr)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        # ------------------
        # Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr, heatmap_1, heatmap_2, heatmap_3 = generator(imgs_lr)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # l2 loss
        loss_l2 = criterion_GAN(gen_hr, imgs_hr)

        # # # Color loss
        # loss_color = lab_color_loss(gen_hr, imgs_hr)
        #
        # Content loss
        f1_2, f2_2, f3_4, f4_4, f5_4 = feature_extractor(gen_hr)
        h1_2, h2_2, h3_4, h4_4, h5_4 = feature_extractor(imgs_hr)

        loss_content = criterion_content(f1_2, h1_2.detach()) + criterion_content(f2_2, h2_2.detach()) + criterion_content(
            f3_4, h3_4.detach()) + criterion_content(f4_4, h4_4.detach()) + criterion_content(f5_4, h5_4.detach())


        # landmark heatmap loss
        real_heatmap = FAN_heatmap(imgs_hr256)
        #print(real_heatmap.shape)

        loss_land_1 = criterion_GAN(heatmap_1, real_heatmap)
        loss_land_2 = criterion_GAN(heatmap_2, real_heatmap)
        loss_land_3 = criterion_GAN(heatmap_3, real_heatmap)

        loss_land_all = loss_land_1 + loss_land_2 + loss_land_3


        # landmark heatmap loss
        imgs_sr256 = uplr_2(gen_hr)
        pre_heatmap = FAN_heatmap(imgs_sr256)

        loss_face_landmark = criterion_GAN(pre_heatmap, real_heatmap)


        ## ideneity loss
        loss_id = arcface_id_loss(gen_hr, imgs_hr)

        ##
        #weight = 1e-2
        #weight_adapt = adjust_learning_rate(epoch, weight)

        # Total loss
        #loss_G = loss_l2 + (1e-2) * loss_color + (1e-2) * loss_content + weight_adapt * loss_GAN + loss_land_all

        loss_G = loss_l2 + loss_land_all + loss_face_landmark + (1e-2) * loss_content + (1e-5) * loss_id + (1e-2) * loss_GAN

        loss_G.backward()
        optimizer_G.step()


        #
        # # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # ---------------------
        #  Results
        # ---------------------

        curr_progress = "[Epoch {:d}/{:d}] [Batch {:d}/{:d}] [MSE loss: {:f}][land loss: {:f}][per loss: {:f}][identity loss: {:f}][gan loss: {:f}]".format(
          epoch, 400, i, len(train_loader), loss_l2.item(), loss_land_all.item(), loss_content.item(), loss_id.item(), loss_GAN.item())
        print(curr_progress)

        # save log
        with open("my_log.txt", 'a') as f:
            f.write(curr_progress + "\n")

        # save results
        batches_done = epoch * len(train_loader) + i
        if batches_done % opt.sample_interval == 0:
            # Save image sample
            save_image(torch.cat((input_lrup.data, gen_hr.data, imgs_hr.data), -2),
                       'images/%d.png' % batches_done, normalize=True)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % epoch)
            #torch.save(discriminator.state_dict(), 'saved_models/discriminator_%d.pth' % epoch)
