#!/bin/bash

#SBATCH --job-name=ARLR
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
python ARLR_run.py $1 $2 $3
