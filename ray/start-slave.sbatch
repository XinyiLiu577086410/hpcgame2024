#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=C064M0256G
#SBATCH --qos=normal
#SBATCH -J raySlave
#SBATCH --nodes=3
#SBATCH --cpus-per-task=8

module load python/hpcgame

export ADDR=$(cat ray-addr.txt)

echo "$(hostname -i): Ray head node address: $ADDR"

srun /lustre/toolchain/conda/envs/hpcgame/bin/ray start --address=$ADDR --num-cpus 4 --block