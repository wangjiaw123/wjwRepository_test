#!/bin/bash
#BATCH -J test_serial
#SBATCH -p cpu-low
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 150:00:00
#SBATCH -o test_serial.out
#SBATCH -e test_serial.err
module load anacoda3
source activate testenv
python /gs/home/sy1909131/HPC_test/test_serial.py

