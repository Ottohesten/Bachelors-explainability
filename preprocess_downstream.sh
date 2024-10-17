#!/bin/bash

#SBATCH --job-name=mmidb_noica_noica_preprocess_5.0
#SBATCH --output=/home/s194101/Bachelors-explainability/experiment_logs/mmidb_noica_noica_preprocess_5.0-%J.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=12gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=otto@skytop.dk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=titans
#SBATCH --export=ALL
#SBATCH --time=04:00:00

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
source activate myenv
python scripts/preprocess_downstream.py --config configs/preprocess_downstream/preprocess_downstream_mmidb_noica_titans.yaml
echo "Done: $(date +%F-%R:%S)


