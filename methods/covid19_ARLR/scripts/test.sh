#!/bin/bash
for ((i = 0 ; i <= 5000 ; i++)); do
      sbatch test.sbatch
done
