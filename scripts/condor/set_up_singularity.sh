#!/bin/bash

USERNAME=$1

cd /scratch/cluster/$USERNAME

mkdir -p tmp/singularity
mkdir -p cache/singularity

TMPDIR=/scratch/cluster/$USERNAME/tmp/singularity SINGULARITY_CACHEDIR=/scratch/cluster/$USERNAME/cache/singularity singularity build -s uniter docker://chenrocks/uniter
TMPDIR=/scratch/cluster/$USERNAME/tmp/singularity SINGULARITY_CACHEDIR=/scratch/cluster/$USERNAME/cache/singularity singularity build -s uniter_butd docker://chenrocks/butd-caffe:nlvr2

mkdir ./uniter/data
mkdir ./uniter_butd/img
mkdir ./uniter_butd/output
