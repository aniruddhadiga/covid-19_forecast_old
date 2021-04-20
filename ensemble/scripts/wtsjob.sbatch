#!/bin/bash

#SBATCH --job-name=ablation
#SBATCH -n 1
#SBATCH --partition=bii
#SBATCH --account=bii_nssac
##SBATCH --reservation=maintenance
#SBATCH --qos=bii-unlimited
#SBATCH --output=log
##SBATCH --time=00:20:00

export CONDA_ROOT="/home/$USER/miniconda3"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate fct
python bma.py $1
