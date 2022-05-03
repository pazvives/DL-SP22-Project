#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_2022sp_16

## change the partition number to use different number of GPUs
##SBATCH --partition=n1s8-v100-1
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8

##SBATCH --partition=n1scans16-v100-2
##SBATCH --gres=gpu:2
##SBATCH --cpus-per-task=16
#SBATCH --partition=n1c24m128-v100-4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16

#SBATCH --time=24:00:00
#SBATCH --output=moco_%j.out
#SBATCH --error=moco_%j.err
#SBATCH --exclusive
#SBATCH --requeue

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

##cp -rp  /scratch/wy547/DL22SP/train /tmp
##cp -rp /scratch/DL22SP/unlabeled_224.sqsh /tmp

##cp -rp /scratch/DL22SP/labeled.sqsh /tmp
echo "Dataset is copied to /tmp"

##--overlay /tmp/unlabeled_224.sqsh \
##--overlay /tmp/labeled.sqsh \
singularity exec --nv \
--bind /scratch \
--overlay /scratch/DL22SP/conda.ext3:ro \
/share/apps/images/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate
##python demo.py
python main_moco.py \
-a resnet50 \
--lr 0.015 \
--batch-size 128 \
--epochs 100 \
--dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
--mlp --moco-t 0.2 --aug-plus --cos \
scratch/$USER/DL-SP22-Project
"
