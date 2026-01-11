# Unaligned Face Super-Resolution (HS-STNnet)
 
Dear friends, Thank you for keep tracking in this implementation of HS-STNnet!


# Installation:

## Clone this repo.

git clone https://github.com/SEU-yang/Unaligned-face-super-resolution.git
cd Unaligned-face-super-resolution/


## Create the anaconda environment by

conda env create -f environment.yml


# Dataset Preparation:

## Our unaligned/aligned face datasets:

[下载训练数据集](https://pan.baidu.com/s/1BvSqXGNz0A_AIlnI5vCPZw?pwd=1234)

提取码: 1234 


## Our extreme_unaligned/aligned face datasets:

[下载extreme_unaligned训练数据集](https://pan.baidu.com/s/19-_Bq2p-D20GG_nIp15PVA?pwd=1234)

提取码: 1234 


## Preparing your own training dataset:

python Unaligned_Faces_Generation.py

Update img_dir and landmark_file to point to your dataset.



# Train New Models:

To train a new model, replace training_list.txt with a file listing the paths of your own low-resolution (LR) and high-resolution (HR) face images. Then run:

python train.py

The models will be saved at ./saved_models


# Test:

To evaluate your trained model, create testing_list.txt containing the paths of your LR and HR face images. Then run:

python test.py


# Evaluiton:

Please refer to the [IQA-PyTorch] (https://github.com/chaofengc/IQA-PyTorch) and [pytorch-fid] (https://github.com/mseitzer/pytorch-fid).


## The code is released for academic research use only.



