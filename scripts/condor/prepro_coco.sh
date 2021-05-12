#!/bin/bash

USERNAME=$1

set -ex

for SPLIT in "train" "val"; do
    # Text DB
    singularity exec -B ./data/:/data,./UNITER:/src \
        --nv -w /scratch/cluster/${USERNAME}/uniter bash -c \
        "cd /src && python3 prepro.py --annotations /data/coco/${SPLIT}.json \
            --output /data/uniter/coco/${SPLIT}/text \
            --task coco"
done
