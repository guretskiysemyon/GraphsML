#!/bin/bash
# Experiment 3 rung C (no-message-passing MLP): tuning with the same protocol and effort as PI-GNN.
#SBATCH --job-name=C_tune1
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=0-12:00:00
#SBATCH --array=0-23
#SBATCH --exclude=s-002
#SBATCH --requeue
#SBATCH --chdir=/home/morg/NLP_2526b/semyong/GraphsML
#SBATCH --output=results/logs/C_tune1_%A_%a.out
#SBATCH --error=results/logs/C_tune1_%A_%a.out

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID"
/home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python -u scripts/tune_stage1.py \
    --task $SLURM_ARRAY_TASK_ID --n-tasks 24 --device cuda --arch mlp --out results/tuning/C_stage1
