#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=C064M0256G
#SBATCH --qos=normal
#SBATCH -J rayWork
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

module load python/hpcgame
export RAY_CLUSTER_ADDR=$(cat ./ray-addr.txt)
python main.py