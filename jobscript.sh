#!/bin/bash

#SBATCH --job-name=BscThesisDRLUpperArm
#SBATCH --time=10:00:00
#SBATCH --mem=5GB

module purge
module load Miniconda3/4.8.3
source $(conda info --base)/etc/profile.d/conda.sh

if conda info --envs | grep -q '^Bsc_Env[[:space:]]'; then
    conda deactivate 2>/dev/null
    conda remove -y -n Bsc_Env --all
fi

conda create -y -n Bsc_Env python=3.10 pip
conda activate Bsc_Env
conda install -c opensim-org -c conda-forge opensim=4.4 simbody

pip install --upgrade pip wheel
pip install -r requirements.txt

python run_script.py --n_envs 1