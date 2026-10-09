#!/bin/bash
# Step H main sweep: 68 DIMACS graphs (proven chi >= 3) x 5 seeds x 100,000 epochs, k = chi - 1.
# One array task = one graph (row = task id in configs/sweep_manifest.csv, smallest n first);
# its 5 seeds run as 5 parallel processes on the task's single GPU.
# Safe to resubmit or requeue: finished runs are skipped, interrupted ones resume.
# Extension in place (decision 13): resubmit a subset with a larger budget, e.g.
#   sbatch --array=18,23 --export=ALL,T_MAX=300000 scripts/run_sweep.sbatch
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

T_MAX=${T_MAX:-100000}
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID restarts=${SLURM_RESTART_COUNT:-0} t_max=$T_MAX"
/home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python -u scripts/run_sweep.py \
    --task-id $SLURM_ARRAY_TASK_ID --seeds 0 1 2 3 4 --t-max $T_MAX --device cuda --out results/main
