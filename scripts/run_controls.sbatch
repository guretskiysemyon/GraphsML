#!/bin/bash
# Step G controls: random / fixed-k DSATUR / randomised DSATUR -> strict vertex descent, k = chi - 1,
# 68 graphs x 3 methods x 5 seeds. One array task = one graph (row = task id in
# configs/sweep_manifest.csv). CPU only, no GPU requested, so any node in the partition can run it.
# Safe to resubmit or requeue: graphs with a DONE marker are skipped.
#SBATCH --job-name=controls
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=0-00:45:00
#SBATCH --array=0-67
#SBATCH --exclude=s-002
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --chdir=/home/morg/NLP_2526b/semyong/GraphsML
#SBATCH --output=results/logs/controls_%A_%a.out
#SBATCH --error=results/logs/controls_%A_%a.out

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID restarts=${SLURM_RESTART_COUNT:-0}"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python -u \
    scripts/run_controls.py --task-id $SLURM_ARRAY_TASK_ID --seeds 0 1 2 3 4 --out results/controls
