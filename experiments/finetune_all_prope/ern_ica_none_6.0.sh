#!/bin/bash

#SBATCH --partition=titans
#SBATCH --job-name=ern_ica_none_6.0
#SBATCH --output=/home/agjma/EEGatScale/experiment_logs/ern_ica_none_6.0-%J.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=12gb
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --time=04:00:00

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
module load CUDA/12.1 CUDNN/8.9
conda activate EEGatScale
python scripts/finetune_cv.py --config configs/finetune_all_prope/ern_ica_none_6.0.yaml

echo "Done: $(date +%F-%R:%S)