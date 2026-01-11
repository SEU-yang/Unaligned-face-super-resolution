# Unaligned Face Super-Resolution (HS-STNnet)
 
Dear friends, Thank you for keep tracking in this implementation of HS-STNnet!


##  Installation:

#Clone this repo.

git clone https://github.com/SEU-yang/Unaligned-face-super-resolution.git
cd Unaligned-face-super-resolution/


#Create the anaconda environment by

conda env create -f environment.yml



##  Dataset Preparation:


#  Our unaligned/aligned face datasets:

[下载训练数据集](https://pan.baidu.com/s/1BvSqXGNz0A_AIlnI5vCPZw?pwd=1234)

提取码: 1234 

#  Our extreme_unaligned/aligned face datasets:

[下载extreme_unaligned训练数据集](https://pan.baidu.com/s/19-_Bq2p-D20GG_nIp15PVA?pwd=1234)

提取码: 1234 


#  Preparing your own training dataset:

python Unaligned_Faces_Generation.py

Change the img_dir and landmark_file to your dataset.



## Train:

python train.py



## Test:

python test.py




