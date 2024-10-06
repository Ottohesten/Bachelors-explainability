#!/bin/bash

#SBATCH --partition=titans
#SBATCH --job-name=pretrain_ica_standardize_downstream
#SBATCH --output=/home/agjma/EEGatScale/experiment_logs/pretrain_ica_standardize_downstream_.0-%J.out
#SBATCH --cpus-per-task=12
#SBATCH --mem=64gb
#SBATCH --gres=gpu:Ampere:3
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --exclude=comp-gpu14
##SBATCH --nodelist=comp-gpu14

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
module load CUDA/12.2 CUDNN/8.9
conda activate EEGatScale
python scripts/pretrain.py --config configs/pretrain_ica_standardize_downstream.yaml

echo "Done: $(date +%F-%R:%S)"