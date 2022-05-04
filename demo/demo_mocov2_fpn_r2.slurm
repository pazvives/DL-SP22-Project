#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_2022sp_16

## change the partition number to use different number of GPUs
##SBATCH --partition=n1s8-v100-1
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8

#SBATCH --partition=n1s16-v100-2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

##SBATCH --partition=n1c24m128-v100-4
##SBATCH --gres=gpu:4
##SBATCH --cpus-per-task=8

##SBATCH --partition=interactive

#SBATCH --time=24:00:00
#SBATCH --output=demo_%j.out
#SBATCH --error=demo_%j.err
#SBATCH --exclusive
#SBATCH --requeue

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

##cp -rp /scratch/DL22SP/unlabeled_224.sqsh /tmp
cp -rp /scratch/DL22SP/labeled.sqsh /tmp
echo "Dataset is copied to /tmp"

##--overlay /tmp/unlabeled_224.sqsh \

singularity exec --nv \
--bind /scratch \
--overlay /scratch/DL22SP/conda.ext3:ro \
--overlay /tmp/labeled.sqsh \
/share/apps/images/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate
python demo_mocov2_fpn.py \
--bp checkpoints/checkpoint_0100.pth.tar \
--resume checkpoint_0022.pth.tar \
--transforms_version v1 \
--lr 0.003 \
--batch-size 4 \
--multiprocessing-distributed \
--world-size 1 --rank 0 \
--epochs 27
"

