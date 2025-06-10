#!/bin/bash
#SBATCH --job-name=salsa_train
#SBATCH --output=salsa_train.out
#SBATCH --error=salsa_train.err
#SBATCH -p volta-gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00:00
#SBATCH --mem=128G
#SBATCH --qos=gpu_access
#SBATCH --mail-type=end
#SBATCH --mail-user=shuhang@unc.edu

# Activate conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate salsa

module load cuda/12.5
#echo "Allocated CUDA devices: $SLURM_JOB_GPUS"

python chef.py
