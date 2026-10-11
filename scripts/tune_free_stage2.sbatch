#!/bin/bash
# Experiment 3: free-logit tuning, stage 2 (same protocol/effort as PI-GNN's tune_stage2.sbatch).
# Needs results/tuning/free_stage2_input_configs.json (tune_free_logits.py --write-stage2).
#SBATCH --job-name=free_tune2
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
#SBATCH --output=results/logs/free_tune2_%A_%a.out
#SBATCH --error=results/logs/free_tune2_%A_%a.out

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID"
/home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python -u scripts/tune_free_logits.py \
    --task $SLURM_ARRAY_TASK_ID --n-tasks 24 --device cuda \
    --configs results/tuning/free_stage2_input_configs.json --n-epochs 50000 --eval-every 1000 --seeds 3 \
    --out results/tuning/free_stage2
