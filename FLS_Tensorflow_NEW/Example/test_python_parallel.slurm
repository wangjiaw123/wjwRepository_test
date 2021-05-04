#!/bin/bash
#BATCH -J test_parallel
#SBATCH -p cpu-low
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 72:00:00
#SBATCH -o test_parallel.out
#SBATCH -e test_parallel.err
module load anacoda3
source activate testenv
python /gs/home/sy1909131/FLS_Tensorflow_NEW/Example/test_parallel.py

