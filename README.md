# Unaligned Face Hallucination with Hierarchical Structure-Aware Spatial Transforming Network


Dear friends, thank you for your interest in this implementation of **HS-STNet** for unaligned face hallucination (face super-resolution).

This repository provides the official code for model training, testing, and dataset preparation.
 


## Directory and File Overview


- **`arcface/`** 
Contains pretrained face recognition models (ArcFace), used to support the identity-preserving loss.

- **`models/`**  Implements the proposed HS-STNet architecture, including the main network, alignment and enhancement modules, and other core components.

- **`Unaligned_Faces_Generation.py`** 
Script for generating aligned and unaligned face image pairs from raw datasets.

- **`train.py`** 
Main training script for model optimization and checkpoint saving.

- **`test.py`**
Script for evaluating the trained model on test images.

- **`environment.yml`** 
Conda environment configuration file for dependency installation and reproducibility.

- **`training_list.txt`**  List of training samples containing paired unaligned low-resolution (LR) and aligned high-resolution (HR) face images.

- **`training_list_all.txt`**  List of extreme unaligned training samples containing paired extreme unaligned LR and aligned HR face images.

- **`README.md`** 
Project documentation and usage instructions.



## Environment


- **OS**: Ubuntu 20.04  
- **GPU**: NVIDIA RTX 4090D  
- **CUDA**: 12.1  
- **PyTorch**: 2.2.2  




## Installation:

Clone this repo.

```bash
git clone https://github.com/SEU-yang/Unaligned-face-super-resolution.git
cd Unaligned-face-super-resolution/
```

## Create the anaconda environment by

```bash
conda env create -f environment.yml
```




## Resources:

### Dataset Preparation:

#### Our unaligned/aligned face datasets:

[BaiduPan: Download the training dataset](https://pan.baidu.com/s/1BvSqXGNz0A_AIlnI5vCPZw?pwd=1234)      (pw: 1234) 

[Google Drive: Download the training dataset](https://drive.google.com/drive/folders/1KCkrRzzLEGWbAPGNFlUBS_ZpQfDNDf86?usp=sharing)



#### Our extreme_unaligned/aligned face datasets:

[BaiduPan: Download the extreme_unaligned training dataset](https://pan.baidu.com/s/19-_Bq2p-D20GG_nIp15PVA?pwd=1234)       (pw: 1234) 


[Google Drive: Download the extreme_unaligned training dataset](https://drive.google.com/drive/folders/1X4H3IAn2Z2JFpDyTYuZyjkdtWc7t40aE?usp=sharing)


#### Preparing your own training dataset:

```bash
python Unaligned_Faces_Generation.py
```

Update img_dir and landmark_file to point to your dataset.



### Train New Models:

To train a new model, replace training_list.txt with a file listing the paths of your own LR and HR face images. Then run:

```bash
python train.py
```

The models will be saved at ./saved_models


### Test:

[BaiduPan: Download the testing dataset](https://pan.baidu.com/s/1LZgbXjinTsuVDInfF-WjgA?pwd=1234)      (pw: 1234) 


[Google Drive: Download the testing dataset](https://drive.google.com/drive/folders/1AMkD6vAx9bO9fQ4zD7gyWEbIT3L3cVJV?usp=sharing)



To evaluate your trained model, create testing_list.txt containing the paths of your LR and HR face images. Then run:

```bash
python test.py

```


## Evaluation:

Please refer to the [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) and [pytorch-fid](https://github.com/mseitzer/pytorch-fid).


## Responsible-use statement:

HS-STNet is designed for academic research on face image super-resolution and face alignment.
Given the sensitive nature of facial data, users are advised to respect privacy and ethical considerations, particularly when applying the model to real-world images.
Potential risks include unauthorized reconstruction of personal identities or misuse in surveillance contexts.
We encourage responsible application of the model, in compliance with applicable regulations and institutional guidelines.



