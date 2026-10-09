#!/bin/bash
# Step G analysis (CPU only): PI-GNN vs random / DSATUR controls after strict vertex descent.
# No GPU requested, so it can start on any node of the partition.
#SBATCH --job-name=analyze_controls
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-00:20:00
#SBATCH --exclude=s-002
#SBATCH --requeue
#SBATCH --chdir=/home/morg/NLP_2526b/semyong/GraphsML
#SBATCH --output=results/logs/analyze_controls_%j.out
#SBATCH --error=results/logs/analyze_controls_%j.out

echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date +%T)"
/home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python -u scripts/analyze_controls.py --out results/analysis
echo "exit code $? end=$(date +%T)"
