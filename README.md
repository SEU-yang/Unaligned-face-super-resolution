# Unaligned Face Hallucination with Hierarchical Structure-Aware Spatial Transforming Network


Dear friends, thank you for your interest in this implementation of **HS-STNet** for unaligned face hallucination (face super-resolution).

This repository provides the official code for model training, testing, and dataset preparation.
 



## Directory and File Overview

#### `arcface/'
Contains face recognition and alignment related modules based on ArcFace, used to support facial feature extraction and alignment.

#### `models/'
Implements the proposed HS-STNet architecture, including alignment and enhancement modules, and other core network components.

#### `Unaligned_Faces_Generation.py'
Script for generating aligned/unaligned face image pairs and corresponding training lists from raw datasets.

#### `train.py'
Main training script for model optimization and checkpoint saving.

#### `test.py'
Script for model inference and evaluation on test images.

#### `environment.yml'
Conda environment configuration file for dependency installation and reproducibility.

#### `training_list.txt'
Training data list containing paired low-resolution and high-resolution face images.

#### `training_list_all.txt'
Extended training list including all available training samples.

#### `README.md'
Project documentation and usage instructions.



## Environment


- **OS**: Ubuntu 20.04  
- **GPU**: NVIDIA RTX 4090D  
- **CUDA**: 12.1  
- **PyTorch**: 2.2.2  



## Installation:

Clone this repo.

`git clone https://github.com/SEU-yang/Unaligned-face-super-resolution.git`

`cd Unaligned-face-super-resolution/`


## Create the anaconda environment by

`conda env create -f environment.yml`


## Dataset Preparation:

### Our unaligned/aligned face datasets:

[下载训练数据集](https://pan.baidu.com/s/1BvSqXGNz0A_AIlnI5vCPZw?pwd=1234)

提取码: 1234 


### Our extreme_unaligned/aligned face datasets:

[下载extreme_unaligned训练数据集](https://pan.baidu.com/s/19-_Bq2p-D20GG_nIp15PVA?pwd=1234)

提取码: 1234 


### Preparing your own training dataset:

`python Unaligned_Faces_Generation.py`

Update img_dir and landmark_file to point to your dataset.



## Train New Models:

To train a new model, replace training_list.txt with a file listing the paths of your own low-resolution (LR) and high-resolution (HR) face images. Then run:

`python train.py`

The models will be saved at ./saved_models


## Test:

To evaluate your trained model, create testing_list.txt containing the paths of your LR and HR face images. Then run:

`python test.py`


## Evaluaiton:

Please refer to the [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) and [pytorch-fid](https://github.com/mseitzer/pytorch-fid).


## Responsible-use statement:

HS-STNet is designed for academic research on face image super-resolution and face alignment.
Given the sensitive nature of facial data, users are advised to respect privacy and ethical considerations, particularly when applying the model to real-world images.
Potential risks include unauthorized reconstruction of personal identities or misuse in surveillance contexts.
We encourage responsible application of the model, in compliance with applicable regulations and institutional guidelines.



