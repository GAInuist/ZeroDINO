#!/bin/bash

# ============================
# ZeroDINO - GZSL Training Script
# Run one dataset at a time.
# ============================

DATASET=$1  # usage: bash train.sh [CUB|SUN|AWA2]

if [ "$DATASET" == "CUB" ]; then
    python train.py --pretrain_path checkpoint/CUB/8336/CUB_ZSLExperiment_0_79.22.pth \
                    --DATASET CUB \
                    --DATASET_path /root/autodl-tmp/dataset/CUB_200_2011/CUB_200_2011/images/ \
                    --attr_num 312


elif [ "$DATASET" == "SUN" ]; then
    python train.py --pretrain_path checkpoint/SUN/4041/SUN_ZSLExperiment_0_60.57.pth \
                    --DATASET SUN \
                    --DATASET_path /root/autodl-tmp/dataset/SUN/images/ \
                    --attr_num 102

elif [ "$DATASET" == "AWA2" ]; then
    python train.py --pretrain_path checkpoint/AWA2/4508/AWA2_ZSLExperiment_0_74.42.pth \
                    --DATASET AWA2 \
                    --DATASET_path /root/autodl-tmp/dataset/AWA2/JPEGImages/ \
                    --attr_num 85
else
    echo "Please choose: CUB | SUN | AWA2"
fi