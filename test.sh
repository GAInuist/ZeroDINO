#!/bin/bash

# ============================
# ZeroDINO - GZSL Testing Script
# Run one dataset at a time.
# ============================

DATASET=$1  # usage: bash test.sh [CUB|SUN|AWA2]

if [ "$DATASET" == "CUB" ]; then
    python local_main.py \
        --config configs/CUB_test.json

elif [ "$DATASET" == "SUN" ]; then
    python local_main.py \
        --config configs/SUN_test.json


elif [ "$DATASET" == "AWA2" ]; then
    python local_main.py \
        --config configs/AWA2_test.json

else
    echo "Usage: bash test.sh [CUB | SUN | AWA2]"
fi