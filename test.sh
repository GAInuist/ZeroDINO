#!/bin/bash

# ============================
# ZeroDINO - GZSL Testing Script
# Run one dataset at a time.
# ============================

DATASET=$1  # usage: bash test.sh [CUB|SUN|AWA2]

if [ "$DATASET" == "CUB" ]; then
    python train.py \
        --pretrain_path checkpoint/CUB_ZeroDINO_79.22.pth \
        --DATASET CUB \
        --DATASET_path /root/autodl-tmp/dataset/CUB_200_2011/CUB_200_2011/images/ \
        --attr_num 312 \
        --is_test True

elif [ "$DATASET" == "SUN" ]; then
    python train.py \
        --pretrain_path checkpoint/SUN_ZeroDINO_60.57.pth \
        --DATASET SUN \
        --DATASET_path /root/autodl-tmp/dataset/SUN/images/ \
        --attr_num 102 \
        --is_test True

elif [ "$DATASET" == "AWA2" ]; then
    python train.py \
        --pretrain_path checkpoint/AWA2_ZeroDINO_74.42.pth \
        --DATASET AWA2 \
        --DATASET_path /root/autodl-tmp/dataset/AWA2/JPEGImages/ \
        --attr_num 85 \
        --is_test True

else
    echo "Usage: bash test.sh [CUB | SUN | AWA2]"
fi