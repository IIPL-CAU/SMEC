#!bin/bash
# Paths 
RAF_PATH=/mnt/c/Erasing-Attention-Consistency/RAF
RESNET_PATH=/mnt/c/Erasing-Attention-Consistency/resnet50_ft_weight.pkl
SAVE_PATH=/mnt/c/Erasing-Attention-Consistency

# Project name : change every experiments! 
PROJECT=puzzle_baseline
GPU_ID=0 # gpu id 

# hyperparams 
BATCH_SIZE=8

python main.py \
    --wandb=${PROJECT} \
    --raf_path=${RAF_PATH} \
    --resnet50_path=${RESNET_PATH} \
    --save_path=${SAVE_PATH} \
    --batch_size=$BATCH_SIZE \
    --gpu=$GPU_ID

# python main.py --label_path 'noise01.txt' --gpu 0
# python main.py --label_path 'noise02.txt'
# python main.py --label_path 'noise03_plus1.txt'

