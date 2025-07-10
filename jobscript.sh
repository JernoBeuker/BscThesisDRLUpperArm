#!/bin/bash

#SBATCH --job-name=BscThesisDRLUpperArm
#SBATCH --time=288:00:00
#SBATCH --mem=5GB

module purge
module load Miniconda3/23.3.1-0

CONDA_ENV_NAME=opensim_env

conda create -y -n $CONDA_ENV_NAME -c conda-forge anaconda python=3.9
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME

pip install --upgrade pip
pip install --upgrade wheel

pip install ./opensim_package/.
pip install -r requirements.txt

echo "Python version: $(python --version)"
echo "Python location: $(which python)"

python run_script.py