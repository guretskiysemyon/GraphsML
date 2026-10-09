#!/bin/bash
# Step I panel C (CPU only): aggregate the soft-space diagnostics from run_soft.py and draw
# figure1_panelC_soft.png. No GPU requested, so it can start on any node of the partition.
# Usually submitted with --dependency=afterany:<run_soft array id>; it prints which runs are missing.
#SBATCH --job-name=analyze_soft
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=0-00:20:00
#SBATCH --exclude=s-002
#SBATCH --requeue
#SBATCH --chdir=/home/morg/NLP_2526b/semyong/GraphsML
#SBATCH --output=results/logs/analyze_soft_%j.out
#SBATCH --error=results/logs/analyze_soft_%j.out

echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date +%T)"
/home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python -u scripts/analyze_soft.py --out results/analysis
echo "exit code $? end=$(date +%T)"
