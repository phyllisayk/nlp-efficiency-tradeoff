#!/bin/sh
#SBATCH --partition=bhuwan
#SBATCH --mem-per-cpu=128G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
 
torchrun --nnodes=1 --nproc_per_node=1 train_model.py allenai/led-base-16384 gov_report 1024 24

