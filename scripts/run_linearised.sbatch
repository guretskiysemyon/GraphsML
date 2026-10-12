#!/bin/bash
# Experiment 3 rung D (PI-GNN linearised at init) on DIMACS, k = chi - 1, step budget 100k, time-matched
# to A. One array task = one graph, 5 seeds as 5 parallel processes on one GPU. Requeue-safe.
#SBATCH --job-name=rungD
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --time=0-04:00:00
#SBATCH --array=0-67
#SBATCH --exclude=s-002
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --chdir=/home/morg/NLP_2526b/semyong/GraphsML
#SBATCH --output=results/logs/rungD_%A_%a.out
#SBATCH --error=results/logs/rungD_%A_%a.out

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID restarts=${SLURM_RESTART_COUNT:-0}"
/home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python -u scripts/run_linearised.py \
    --task-id $SLURM_ARRAY_TASK_ID --seeds 0 1 2 3 4 --t-max 100000 --device cuda --out results/exp3
