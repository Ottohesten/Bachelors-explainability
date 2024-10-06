#!/bin/bash

#SBATCH --partition=cyclopes
#SBATCH --job-name=preprocess_downstream_ern_noica
#SBATCH --output=/home/agjma/EEGatScale/experiment_logs/preprocess_downstream_ern_noica-%J.out
#SBATCH --cpus-per-task=30
#SBATCH --mem=64gb
#SBATCH --export=ALL
#SBATCH --time=02:00:00

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)
"

source ~/.bashrc
module load CUDA/12.1 CUDNN/8.9
conda activate EEGatScale
python scripts/preprocess_downstream.py --config /home/agjma/EEGatScale/configs/preprocess_downstream/preprocess_downstream_ern_noica.yaml

echo "Done: $(date +%F-%R:%S)"