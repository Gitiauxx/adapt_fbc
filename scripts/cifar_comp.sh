#!/bin/bash

#SBATCH --job-name=cifar_comp
#SBATCH --output=/scratch/xgitiaux/cifar_comp_%j.out
#SBATCH --error=/scratch/xgitiaux/cifar_comp_%j.error
#SBATCH --mail-user=xgitiaux@gmu.edu
#SBATCH --mail-type=END
#SBATCH --export=ALL
#SBATCH --partition=gpuq
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --mem=64G
#SBATCH --qos=csqos
#SBATCH --gres=gpu:4
#SBATCH -t 0-10:00

module load python/3.6.7
module load cuda/10.2
source ../fvae-env/bin/activate

echo $SLURM_ARRAY_TASK_ID
../fvae-env/bin/python3 eval.py --config_path configs/cifar/cifar_pareto_comp_cnn.yml --seed 0 --beta 0.0 --tmax 1.0