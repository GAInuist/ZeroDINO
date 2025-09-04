#!/bin/bash

# ============================
# ZeroDINO - GZSL Training Script
# Run one dataset at a time.
# ============================

DATASET=$1  # usage: bash train.sh [CUB|SUN|AWA2]

if [ "$DATASET" == "CUB" ]; then
    python local_main.py \
                    --config configs/CUB_train.json

elif [ "$DATASET" == "SUN" ]; then
    python local_main.py \
                    --config configs/SUN_train.json

elif [ "$DATASET" == "AWA2" ]; then
    python local_main.py \
                    --config configs/AWA2_train.json
else
    echo "Please choose: CUB | SUN | AWA2"
fi