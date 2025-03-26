#!/bin/bash
#SBATCH -J path2_err
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH --ntasks-per-node 48
#SBATCH --cpus-per-task 1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

hostname; pwd; date
ibrun python path2_loo_err.py --data_dir '/scratch/06536/qezlou/Goku/processed_data/xi_bins/'
