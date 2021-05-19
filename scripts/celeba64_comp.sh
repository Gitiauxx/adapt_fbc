#!/bin/bash

#SBATCH --job-name=celeba64_comp
#SBATCH --output=/scratch/xgitiaux/celeba64_comp_%j.out
#SBATCH --error=/scratch/xgitiaux/celeba64_comp_%j.error
#SBATCH --mail-user=xgitiaux@gmu.edu
#SBATCH --mail-type=END
#SBATCH --export=ALL
#SBATCH --partition=gpuq
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --qos=csqos
#SBATCH --gres=gpu:4

module load python/3.8.4
module load cuda/10.2
source ../afbc-env/bin/activate

echo $SLURM_ARRAY_TASK_ID
../fvae-env/bin/python3 eval.py --config_path configs/celeba64/celeba_pareto_comp_cnn.yml --seed 0 --beta 0.0 --tmax 1.0