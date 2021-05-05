#!/bin/bash

USERNAME=$1

set -ex

for SPLIT in "train" "val" "test"; do
    # Text DB
    singularity exec -B ./data/:/data,./UNITER:/src \
        --nv -w /scratch/cluster/${USERNAME}/uniter bash -c \
        "cd /src && python3 prepro.py --annotations /data/vizwiz/Annotations/${SPLIT}.json \
            --output /data/uniter/vizwiz/${SPLIT}/text \
            --task vizwiz"
    if [[ ${SPLIT} == "train" ]]; then
        if [[ -f ./data/vizwiz/Annotations/${SPLIT}_new.json ]]; then
            singularity exec -B ./data/:/data,./UNITER:/src \
                --nv -w /scratch/cluster/${USERNAME}/uniter bash -c \
                "cd /src && python3 prepro.py --annotations /data/vizwiz/Annotations/${SPLIT}_new.json \
                    --output /data/uniter/vizwiz/${SPLIT}/text_new \
                    --task vizwiz"
        fi
        if [[ -f ./data/vizwiz/Annotations/${SPLIT}_val.json ]]; then
            singularity exec -B ./data/:/data,./UNITER:/src \
                --nv -w /scratch/cluster/${USERNAME}/uniter bash -c \
                "cd /src && python3 prepro.py --annotations /data/vizwiz/Annotations/${SPLIT}_val.json \
                    --output /data/uniter/vizwiz/${SPLIT}/text_val \
                    --task vizwiz"
        fi
    fi

    # Image features
    if [[ ! -d ./data/uniter/vizwiz/${SPLIT}/image/all ]]; then
        mkdir -p ./data/uniter/vizwiz/${SPLIT}/image/all
    fi
    singularity exec -B ./data/vizwiz/${SPLIT}:/img,./data/uniter/vizwiz/${SPLIT}/image/all:/output \
        --nv -w /scratch/cluster/${USERNAME}/uniter_butd bash -c \
        "cd /src && python -u tools/generate_npz.py --gpu 0"

    # Image DB
    if [[ ! -d ./data/uniter/vizwiz/${SPLIT}/tmp ]]; then
        mkdir -p ./data/uniter/vizwiz/${SPLIT}/tmp
    fi
    singularity exec -B ./data/:/data,./UNITER:/src \
        --nv -w /scratch/cluster/${USERNAME}/uniter bash -c \
        "cd /src && python3 scripts/convert_imgdir.py --img_dir /data/uniter/vizwiz/${SPLIT}/image/all \
            --output /data/uniter/vizwiz/${SPLIT}/tmp"
    mv ./data/uniter/vizwiz/${SPLIT}/tmp/all/* ./data/uniter/vizwiz/${SPLIT}/image/
    rmdir ./data/uniter/vizwiz/${SPLIT}/tmp/all
    rmdir ./data/uniter/vizwiz/${SPLIT}/tmp
done
