#!bin/bash
# Paths 
RAF_PATH=/HDD/jihyun/RAF_DB/
RESNET_PATH=/home/jihyun/code/eac_puzzle/model/resnet50_ft_weight.pkl
SAVE_PATH=/home/jihyun/code/eac_puzzle/

# Project name : change every experiments! 
PROJECT=puzzle_nogumbel_224_2_04
GPU_ID=0 # gpu id 

# hyperparams 
BATCH_SIZE=32

LABEL_PATH=noise04.txt

python main.py \
    --wandb=${PROJECT} \
    --raf_path=${RAF_PATH} \
    --resnet50_path=${RESNET_PATH} \
    --save_path=${SAVE_PATH} \
    --batch_size=$BATCH_SIZE \
    --gpu=$GPU_ID \
    --label_path=$LABEL_PATH

# python main.py --label_path 'noise01.txt' --gpu 0
# python main.py --label_path 'noise02.txt'
# python main.py --label_path 'noise03_plus1.txt'

