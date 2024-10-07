#!/bin/bash
#BSUB -J pretrain_2
#BSUB -o pretrain_2_%J.out
#BSUB -e pretrain_2_%J.err
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=4G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 08:00
#BSUB -u otto@skytop.dk
#BSUB -B
#BSUB -N
# end of BSUB options

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

module load python3/3.11.9

source env/bin/activate
python scripts/pretrain.py --config configs/pretrain/pretrain_bendr_raw_all.yaml fit

echo "Done: $(date +%F-%R:%S)"
