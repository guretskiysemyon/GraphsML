#!/bin/bash
# Step H main sweep: 68 DIMACS graphs (proven chi >= 3) x 5 seeds x 100,000 epochs, k = chi - 1.
# One array task = one graph (row = task id in configs/sweep_manifest.csv, smallest n first);
# its 5 seeds run as 5 parallel processes on the task's single GPU.
# Safe to resubmit or requeue: finished runs are skipped, interrupted ones resume.
#SBATCH --job-name=pignn_sweep
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=6G
#SBATCH --time=0-01:30:00
#SBATCH --array=0-67
#SBATCH --exclude=s-002
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --chdir=/home/morg/NLP_2526b/semyong/GraphsML
#SBATCH --output=results/logs/sweep_%A_%a.out
#SBATCH --error=results/logs/sweep_%A_%a.out

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID restarts=${SLURM_RESTART_COUNT:-0}"
/home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python -u scripts/run_sweep.py \
    --task-id $SLURM_ARRAY_TASK_ID --seeds 0 1 2 3 4 --t-max 100000 --device cuda --out results/main
