#!/bin/sh

# set up conda environment
conda create --name nlp-bmk python=3.8.12 -y
conda activate nlp-bmk

# download the requirements
conda install pip
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

