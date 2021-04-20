#!/bin/bash

#SBATCH -A biocomplexity 
#SBATCH --time=5:00:00
#SBATCH --nodes=1 --ntasks-per-node=30
#SBATCH -p bii
# #SBATCH -p standard
 
state=$1
echo $state
python ../../script/lstm_ensemble.py -r $state
