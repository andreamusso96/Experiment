#!/bin/bash

#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=01:00:00

module load stack/2024-06 python/3.11.6
source /cluster/home/anmusso/Experiment/experiment_venv/bin/activate

python experiment.py "$@"