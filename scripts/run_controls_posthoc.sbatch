#!/bin/bash
# Post hoc control (added after the step G results; docs/progress_exp1.md section 10 caveats 1-2):
# randomised DSATUR with random vertex ties only and lowest-index colours ('dsatur_rand_v'),
# 68 graphs x 10 seeds -> strict vertex descent, k = chi - 1. Writes controls_posthoc.jsonl and
# DONE_posthoc next to the pre-registered controls.jsonl, which is not touched.
# One array task = one graph (row = task id in configs/sweep_manifest.csv). CPU only.
#SBATCH --job-name=controls_posthoc
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
#SBATCH --output=results/logs/controls_posthoc_%A_%a.out
#SBATCH --error=results/logs/controls_posthoc_%A_%a.out

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID restarts=${SLURM_RESTART_COUNT:-0}"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python -u \
    scripts/run_controls.py --task-id $SLURM_ARRAY_TASK_ID --methods dsatur_rand_v \
    --seeds 0 1 2 3 4 5 6 7 8 9 --tag posthoc --out results/controls
