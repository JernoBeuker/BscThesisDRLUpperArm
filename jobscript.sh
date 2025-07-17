#!/bin/bash

#SBATCH --job-name=BscThesisDRLUpperArm
#SBATCH --time=10-0
#SBATCH --mem=100GB

module purge
module load Miniconda3/4.8.3

conda activate Bsc_Env

python run_script.py --n_envs 1