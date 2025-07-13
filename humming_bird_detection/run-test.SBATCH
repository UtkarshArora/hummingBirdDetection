#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem=25GB
#SBATCH --gres=gpu
#SBATCH --job-name=hbird
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ua2152@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
module purge

singularity exec --nv \
            --overlay my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
            /bin/bash -c "source /ext3/env.sh; python train_v2.py"