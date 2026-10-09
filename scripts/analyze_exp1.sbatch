#!/bin/bash
# Step I analysis (CPU only). No GPU requested, so it can start on any node of the partition.
#SBATCH --job-name=analyze_exp1
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --requeue
#SBATCH --chdir=/home/morg/NLP_2526b/semyong/GraphsML
#SBATCH --output=results/logs/analyze_%j.out
#SBATCH --error=results/logs/analyze_%j.out

echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date +%T)"
/home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python -u scripts/analyze_exp1.py --out results/analysis
echo "end=$(date +%T)"
