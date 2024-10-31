#!/bin/bash

#SBATCH --job-name=mmidb_noica_noica_5.0_nogroups
#SBATCH --output=/home/s194101/Bachelors-explainability/experiment_logs/mmidb_noica_noica_5.0_nogroups-%J.out
#SBATCH --cpus-per-task=10
#SBATCH --mem=100gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=otto@skytop.dk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=titans
#SBATCH --export=ALL
#SBATCH --time=10:00:00

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
source activate myenv
python scripts/finetune_cv_nogroups.py --config configs/finetune/mmidb_noica_noica_5.0_nogroups.yaml

echo "Done: $(date +%F-%R:%S)"
