#!/bin/bash

#SBATCH --job-name=bendr
#SBATCH --output=bendr-%J.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=40gb
#SBATCH --gres=gpu:Ampere:3
#SBATCH --export=ALL
##SBATCH --nodelist=comp-gpu04

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
module load CUDA/12.1 CUDNN/8.9
conda activate eegatscale
python scripts/pretrain.py -c configs/pretrain.yaml fit

echo "Done: $(date +%F-%R:%S)"
