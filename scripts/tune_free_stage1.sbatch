#!/bin/bash
# Experiment 3: free-logit tuning, stage 1 (same protocol/effort as PI-GNN's tune_stage1.sbatch).
#SBATCH --job-name=free_tune1
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=0-12:00:00
#SBATCH --array=0-23
#SBATCH --exclude=s-002
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --chdir=/home/morg/NLP_2526b/semyong/GraphsML
#SBATCH --output=results/logs/free_tune1_%A_%a.out
#SBATCH --error=results/logs/free_tune1_%A_%a.out

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID"
/home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python -u scripts/tune_free_logits.py \
    --task $SLURM_ARRAY_TASK_ID --n-tasks 24 --device cuda --out results/tuning/free_stage1
