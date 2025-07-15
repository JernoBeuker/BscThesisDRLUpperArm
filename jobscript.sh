#!/bin/bash

#SBATCH --job-name=BscThesisDRLUpperArm
#SBATCH --time=1-0
#SBATCH --mem=5GB

module purge
module load Miniconda3/4.8.3

source activate Bsc_Env

python test_environment.py