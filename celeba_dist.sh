#!/bin/bash
#SBATCH --job-name=celeba_dist
#SBATCH --partition=gpuq
#SBATCH --time=72:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node

#SBATCH --job-name=celeba64_tcomp
#SBATCH --output=/scratch/xgitiaux/celeba64_dist_%j.out
#SBATCH --error=/scratch/xgitiaux/celeba64_dist_%j.error
#SBATCH --mail-user=xgitiaux@gmu.edu
#SBATCH --mail-type=END
#SBATCH --export=ALL
#SBATCH --partition=gpuq
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-gpu=8
#SBATCH --qos=csqos
#SBATCH --gres=gpu:1

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=2
#
#### get the first node name as master address - customized for vgg slurm
#### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}

#if [ ${SLURM_NODELIST:7:1} == "," ]; then
#    echo "MASTER_ADDR="${SLURM_NODELIST:0:7}
#    export MASTER_ADDR=${SLURM_NODELIST:0:7}
#elif [ ${SLURM_NODELIST:6:1} == "[" ]; then
#    echo "MASTER_ADDR="${SLURM_NODELIST:0:6}${SLURM_NODELIST:7:1}
#    export MASTER_ADDR=${SLURM_NODELIST:0:6}${SLURM_NODELIST:7:1}
#else
#    echo "MASTER_ADDR="${SLURM_NODELIST}
#    export MASTER_ADDR=${SLURM_NODELIST}
#fi

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

module load python/3.8.4
module load cuda/10.2
source ../afbc-env/bin/activate

echo $SLURM_ARRAY_TASK_ID
../afbc-env/bin/python3 train_dist.py --path ../data_vae