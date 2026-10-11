#!/bin/bash
# Experiment 3 always-logged diagnostics (4C Dirichlet energy, 4E movement from init) from saved
# checkpoints. CPU only. One array task = one graph, all 5 seeds. Which arm: RUNS in the environment.
#   sbatch scripts/run_movement.sbatch                                   # arm A (results/main)
#   sbatch --export=ALL,RUNS=results/exp3/B scripts/run_movement.sbatch  # arm B
#SBATCH --job-name=movement
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-01:00:00
#SBATCH --array=0-67
#SBATCH --exclude=s-002
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --chdir=/home/morg/NLP_2526b/semyong/GraphsML
#SBATCH --output=results/logs/movement_%A_%a.out
#SBATCH --error=results/logs/movement_%A_%a.out

RUNS=${RUNS:-results/main}
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID runs=$RUNS"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python -u \
    scripts/run_movement.py --task-id $SLURM_ARRAY_TASK_ID --runs $RUNS
