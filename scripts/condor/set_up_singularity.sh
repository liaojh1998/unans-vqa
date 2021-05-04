#!/bin/bash

USERNAME=$1

cd /scratch/cluster/$USERNAME

mkdir -p tmp/singularity
mkdir -p cache/singularity

TMPDIR=/scratch/cluster/$USERNAME/tmp/singularity SINGULARITY_CACHEDIR=/scratch/cluster/$USERNAME/cache/singularity singularity build -s uniter docker://chenrocks/uniter
TMPDIR=/scratch/cluster/$USERNAME/tmp/singularity SINGULARITY_CACHEDIR=/scratch/cluster/$USERNAME/cache/singularity singularity build -s uniter_butd docker://chenrocks/butd-caffe:nlvr2

# Set up directories to mount
mkdir ./uniter/data
mkdir ./uniter_butd/img
mkdir ./uniter_butd/output

# Modify buggy generate_npz.py code
cd -
cp ./utils/generate_npz.py /scratch/cluster/$USERNAME/uniter_butd/src/tools/
