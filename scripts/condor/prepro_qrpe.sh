#!/bin/bash

USERNAME=$1

set -ex

for SPLIT in "train" "val"; do
    # Text DB
    singularity exec -B ./data/:/data,./UNITER:/src \
        --nv -w /scratch/cluster/${USERNAME}/uniter bash -c \
        "cd /src && python3 prepro.py --annotations /data/qrpe/${SPLIT}.json \
            --output /data/uniter/qrpe/${SPLIT}/text \
            --task qrpe \
            --name ${SPLIT}"
    if [[ ${SPLIT} == "train" ]]; then
        if [[ -f ./data/qrpe/${SPLIT}_new.json ]]; then
            singularity exec -B ./data/:/data,./UNITER:/src \
                --nv -w /scratch/cluster/${USERNAME}/uniter bash -c \
                "cd /src && python3 prepro.py --annotations /data/qrpe/${SPLIT}_new.json \
                    --output /data/uniter/qrpe/${SPLIT}/text_new \
                    --task qrpe \
                    --name ${SPLIT}"
        fi
        if [[ -f ./data/qrpe/${SPLIT}_val.json ]]; then
            singularity exec -B ./data/:/data,./UNITER:/src \
                --nv -w /scratch/cluster/${USERNAME}/uniter bash -c \
                "cd /src && python3 prepro.py --annotations /data/qrpe/${SPLIT}_val.json \
                    --output /data/uniter/qrpe/${SPLIT}/text_val \
                    --task qrpe \
                    --name ${SPLIT}"
        fi
    fi

    # Image features
    if [[ ! -d ./data/uniter/coco/${SPLIT}/image/all ]]; then
        mkdir -p ./data/uniter/coco/${SPLIT}/image/all
    fi
    singularity exec -B ./data/coco/${SPLIT}2014:/img,./data/uniter/coco/${SPLIT}/image/all:/output \
        --nv -w /scratch/cluster/${USERNAME}/uniter_butd bash -c \
        "cd /src && python -u tools/generate_npz.py --gpu 0"

    # Image DB
    if [[ ! -d ./data/uniter/coco/${SPLIT}/tmp ]]; then
        mkdir -p ./data/uniter/coco/${SPLIT}/tmp
    fi
    singularity exec -B ./data/:/data,./UNITER:/src \
        --nv -w /scratch/cluster/${USERNAME}/uniter bash -c \
        "cd /src && python3 scripts/convert_imgdir.py --img_dir /data/uniter/coco/${SPLIT}/image/all \
            --output /data/uniter/coco/${SPLIT}/tmp"
    mv ./data/uniter/coco/${SPLIT}/tmp/all/* ./data/uniter/coco/${SPLIT}/image/
    rmdir ./data/uniter/coco/${SPLIT}/tmp/all
    rmdir ./data/uniter/coco/${SPLIT}/tmp
done
