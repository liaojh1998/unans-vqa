#!/bin/bash

USERNAME=$1
CONFIG=$2

singularity exec -B /scratch/cluster/${USERNAME}/gnlp/unans-vqa/:/src,/scratch/cluster/${USERNAME}/gnlp/unans-vqa/data:/data \
    --nv -w /scratch/cluster/${USERNAME}/uniter bash -c \
    "cd /src/UNITER && python3 train_unans.py --config config/${CONFIG}"
