#!/bin/bash

#SBATCH --job-name=tuev_noica_noica_5.0_nogroups
#SBATCH --output=/home/s194101/Bachelors-explainability/logs/tuev_noica_noica_5.0_nogroups-%J.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=40gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=otto@skytop.dk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=titans
#SBATCH --export=ALL
#SBATCH --time=12:00:00

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
source activate myenv
python scripts/finetune_cv_nogroups.py --config configs/finetune/tuev_noica_noica_5.0_nogroups.yaml

echo "Done: $(date +%F-%R:%S)"
