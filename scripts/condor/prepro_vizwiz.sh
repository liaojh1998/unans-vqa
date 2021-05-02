#!/bin/bash

USERNAME=$1

set -ex

for SPLIT in "train" "val" "test"; do
    if [ ! -d ./data/uniter/vizwiz/${SPLIT}/image ]; then
        mkdir -p ./data/uniter/vizwiz/${SPLIT}/image
    fi

    singularity exec -B ./data/:/data,./UNITER:/src \
        --nv -w /scratch/cluster/${USERNAME}/uniter bash -c \
        "cd /src && python3 prepro.py --annotations /data/vizwiz/Annotations/${SPLIT}.json \
            --output /data/uniter/vizwiz/${SPLIT}/text \
            --task vizwiz"
    singularity exec -B ./data/vizwiz/${SPLIT}:/img,./data/uniter/vizwiz/${SPLIT}/image_feat/:/output \
        --nv -w /scratch/cluster/${USERNAME}/uniter_butd bash -c \
        "cd /src && python tools/generate_npz.py --gpu 0"
    singularity exec -B ./data/:/data,./UNITER:/src \
        --nv -w /scratch/cluster/${USERNAME}/uniter bash -c \
        "cd /src && python3 scripts/convert_imgdir.py --img_dir /data/uniter/vizwiz/${SPLIT}/image_feat \
            --output /data/uniter/vizwiz/${SPLIT}/image"
done
