#!/bin/bash

USERNAME=$1

singularity exec -B /scratch/cluster/${USERNAME}/gnlp/unans-vqa:/src,/scratch/cluster/${USERNAME}/gnlp/unans-vqa/data:/data \
    --nv -w /scratch/cluster/${USERNAME}/uniter bash
