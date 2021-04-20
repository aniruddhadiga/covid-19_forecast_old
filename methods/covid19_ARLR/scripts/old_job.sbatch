#!/bin/bash

#SBATCH --job-name=ARLR
##SBATCH --nodes=1
#SBATCH -n 1
##SBATCH --mem=50gb
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=1
#SBATCH --partition=bii
#SBATCH --account=biocomplexity
#SBATCH --output=log
#SBATCH --time=8:00:00
##SBATCH --exclude=udc-aj37-15c1,udc-aj37-16c1,udc-aj40-16c0,udc-aj40-17c0,udc-aj40-17c1,udc-aj40-18c0,udc-aj37-17c0,udc-aj38-3c0,udc-aj38-9c0
# Below here enter the commands to start your job

#module add Python/2.7.14-foss-2018a
#hostname
export CONDA_ROOT="/home/$USER/miniconda3"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate fct
python ARLR_run.py $1 $2 $3
