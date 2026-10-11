#!/bin/bash
# TabuCol quality reference: 68 graphs x 10 seeds, k = chi - 1, random start, time budget per graph =
# PI-GNN's mean training time to 100k epochs on that graph (~300-700 s). One array task = one graph
# (row = task id in configs/sweep_manifest.csv); its 10 seeds run as 10 single-threaded processes on
# the task's 10 CPUs. CPU only, no GPU requested.
# Safe to resubmit or requeue: finished seeds are skipped, a killed seed restarts from scratch.
# Extra arguments are passed to run_tabucol.py, e.g. for the timing test:
#   sbatch --array=2,67 scripts/run_tabucol.sbatch --time-budget 60 --out results/tabucol_test
#SBATCH --job-name=tabucol
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --time=0-00:45:00
#SBATCH --array=0-67
#SBATCH --exclude=s-002
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --chdir=/home/morg/NLP_2526b/semyong/GraphsML
#SBATCH --output=results/logs/tabucol_%A_%a.out
#SBATCH --error=results/logs/tabucol_%A_%a.out

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID restarts=${SLURM_RESTART_COUNT:-0} args=$*"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python -u \
    scripts/run_tabucol.py --task-id $SLURM_ARRAY_TASK_ID "$@"
