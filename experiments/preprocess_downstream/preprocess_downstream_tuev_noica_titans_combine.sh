#!/bin/bash

#SBATCH --job-name=tuev_noica_noica_preprocess_5.0_titans_combine
#SBATCH --output=/home/s194101/Bachelors-explainability/logs/tuev_noica_noica_preprocess_5.0_titans_combine-%J.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=40gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=otto@skytop.dk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=titans
#SBATCH --export=ALL
#SBATCH --time=08:00:00

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
source activate myenv
python scripts/preprocess_downstream.py --config configs/preprocess_downstream/preprocess_downstream_tuev_noica_titans_combine_test.yaml
echo "Done: $(date +%F-%R:%S)"


