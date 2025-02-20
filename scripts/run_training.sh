#!/bin/bash
#SBATCH --job-name=test_train_genAI
#SBATCH --account=bsc72
#SBATCH --chdir=.
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --qos=acc_bscls

module purge

ml bsc/1.0
ml anaconda
source activate genAI

wandb login 7952be2b9b469177c60dcaee07f53602e3f2f7f3
wandb offline

srun python -u train.py --config config.yaml
