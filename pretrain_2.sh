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

module load python3/3.11.9

source env/bin/activate
python scripts/pretrain.py --config configs/pretrain/pretrain_bendr_raw_all.yaml fit

echo "Done: $(date +%F-%R:%S)"