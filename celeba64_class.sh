#!/bin/bash

#SBATCH --job-name=celeba64_class
#SBATCH --output=/scratch/xgitiaux/celeba64_class_%j.out
#SBATCH --error=/scratch/xgitiaux/celeba64_class_%j.error
#SBATCH --mail-user=xgitiaux@gmu.edu
#SBATCH --mail-type=END
#SBATCH --export=ALL
#SBATCH --partition=gpuq
#SBATCH --nodes 1
##SBATCH --tasks 1
#SBATCH --ntasks-per-node 16
#SBATCH --mem=64G
##SBATCH --cpus-per-gpu=8
#SBATCH --qos=csqos
#SBATCH --gres=gpu:1

module load python/3.8.4
module load cuda/10.2
source ../afbc-env/bin/activate

echo $SLURM_ARRAY_TASK_ID
../afbc-env/bin/python3 train_class.py --path ../data_celeba_tar