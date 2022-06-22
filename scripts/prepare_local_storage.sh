#!/bin/bash
#Creating Folders
cd /content
echo "===> Creating folders..."
mkdir DATASETS/
mkdir DATASETS/ActionTubesV2
mkdir DATASETS/ActionTubesV2Scored
mkdir DATASETS/Pretrained_Models

#Update wget
pip install --upgrade --no-cache-dir gdown

# #Installing yacs
echo "===> Installing yacs"
pip3 install yacs

# #Installing CUDA
echo "===> Installing CUDA"
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

echo "===> Installing pytorchvideo"
pip install pytorchvideo