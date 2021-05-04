#!/bin/bash

singularity exec -B /scratch/cluster/liaojh/gnlp/unans-vqa:/src,/scratch/cluster/liaojh/gnlp/unans-vqa/data:/data \
    --nv -w /scratch/cluster/liaojh/uniter bash
