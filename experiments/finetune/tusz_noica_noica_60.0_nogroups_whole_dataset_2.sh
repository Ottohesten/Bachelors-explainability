#!/bin/bash

#SBATCH --job-name=tusz_noica_noica_60.0_nogroups_whole_dataset_2
#SBATCH --output=/home/s194101/Bachelors-explainability/logs/tusz_noica_noica_60.0_nogroups_whole_dataset_2-%J.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=otto@skytop.dk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=titans
#SBATCH --export=ALL
#SBATCH --time=16:00:00

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
source activate myenv
python scripts/finetune_cv_nogroups.py --config configs/finetune/tusz_noica_noica_60.0_nogroups_whole_dataset_2.yaml

echo "Done: $(date +%F-%R:%S)"
