#!/bin/bash

#SBATCH --job-name=tusz_noica_noica_60.0_nogroups
#SBATCH --output=/home/s194101/Bachelors-explainability/logs/tusz_noica_noica_60.0_nogroups-%J.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=100gb
#SBATCH --gres=gpu:2
#SBATCH --mail-user=otto@skytop.dk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=titans
#SBATCH --export=ALL
#SBATCH --time=24:00:00

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
source activate myenv
python scripts/finetune_cv_nogroups.py --config configs/finetune/tusz_noica_noica_60.0_nogroups.yaml

echo "Done: $(date +%F-%R:%S)"
