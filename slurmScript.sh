#!/bin/bash
#SBATCH --job-name=gpt2-hpo
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/hpo-%j.out
#SBATCH --error=logs/hpo-%j.err

uv run supreme-robot
