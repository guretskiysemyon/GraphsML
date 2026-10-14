#!/bin/bash
# Experiment 2 (RQ3): decoders + null-P controls on frozen P, k = chi - 1, epoch 100k.
# One array task = one run (graph row = task id // 5, seed = task id % 5): 68 graphs x 5 seeds.
# CPU only, no GPU requested. Safe to resubmit or requeue: runs with a DONE marker are skipped.
# Environment overrides (sbatch --export=ALL,RUNS=...,ARM=...,VARIANTS="..."):
#   RUNS      arm directory with the checkpoints   (default results/main)
#   ARM       output name under results/exp2/      (default A)
#   VARIANTS  P variants to decode                 (default: all five)
#SBATCH --job-name=decoders
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=0-01:00:00
#SBATCH --array=0-339
#SBATCH --exclude=s-002
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --chdir=/home/morg/NLP_2526b/semyong/GraphsML
#SBATCH --output=results/logs/decoders_%A_%a.out
#SBATCH --error=results/logs/decoders_%A_%a.out

RUNS=${RUNS:-results/main}
ARM=${ARM:-A}
VARIANTS=${VARIANTS:-trained uniform untrained rowperm colperm}
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID restarts=${SLURM_RESTART_COUNT:-0} runs=$RUNS arm=$ARM start=$(date +%T)"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python -u \
    scripts/run_decoders.py --task-id $SLURM_ARRAY_TASK_ID --runs $RUNS --arm $ARM --variants $VARIANTS
echo "exit=$? end=$(date +%T)"
