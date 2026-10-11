#!/bin/bash
# Experiment 3 (RQ2) free-logit arms on DIMACS, k = chi - 1. One array task = one graph (row of
# configs/sweep_manifest.csv), 5 seeds as 5 parallel processes on one GPU (same layout as arm A).
# Arm and options come from the environment (default ARM=B):
#   sbatch --export=ALL,ARM=B     scripts/run_free_logits.sbatch
#   sbatch --export=ALL,ARM=Brand scripts/run_free_logits.sbatch
#   sbatch --export=ALL,ARM=B,EXTRA="--logit-noise 0.5" scripts/run_free_logits.sbatch   # noise control
# Safe to resubmit or requeue: finished runs are skipped, interrupted ones resume.
#SBATCH --job-name=free_logits
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=6G
#SBATCH --time=0-02:00:00
#SBATCH --array=0-67
#SBATCH --exclude=s-002
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --chdir=/home/morg/NLP_2526b/semyong/GraphsML
#SBATCH --output=results/logs/free_logits_%A_%a.out
#SBATCH --error=results/logs/free_logits_%A_%a.out

ARM=${ARM:-B}
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID restarts=${SLURM_RESTART_COUNT:-0} arm=$ARM extra=$EXTRA"
/home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python -u scripts/run_free_logits.py \
    --task-id $SLURM_ARRAY_TASK_ID --arm $ARM --seeds 0 1 2 3 4 --device cuda --out results/exp3 $EXTRA
