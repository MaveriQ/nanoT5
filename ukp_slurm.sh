#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:00:00
#SBATCH --job-name=morpht5
#SBATCH --output=/storage/ukp/work/jabbar/git/nanoT5/slurm_outputs/gpt2_200k.out
#SBATCH --error=/storage/ukp/work/jabbar/git/nanoT5/slurm_outputs/gpt2_200k.err

echo "starting script"

python /storage/ukp/work/jabbar/git/nanoT5/nanoT5/main.py \
      model.use_gpt2=true 