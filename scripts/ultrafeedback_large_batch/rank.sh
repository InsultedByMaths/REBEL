#!/bin/bash
#SBATCH --job-name=test_rank
#SBATCH --time=24:00:00
# #SBATCH --partition=gpu
#SBATCH --partition kempner_h100
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --mail-user=nianli_peng@g.harvard.edu
#SBATCH -o ./runs/output_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./runs/errors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=begin

module load cuda

python ./src/ultrafeedback_large_batch/rank_all.py