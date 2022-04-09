#!/bin/sh
#SBATCH --partition=bhuwan
#SBATCH --mem-per-cpu=128G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
 
torchrun --nnodes=1 --nproc_per_node=1 train_model.py ./results/led-base-16384_gov_report_1024_bsz24_1gpus_50epochs gov_report 1024 24

