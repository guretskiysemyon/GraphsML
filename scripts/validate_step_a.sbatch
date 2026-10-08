#!/bin/bash
#SBATCH --job-name=pignn_valA
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-10
#SBATCH --chdir=/home/morg/NLP_2526b/semyong/GraphsML
#SBATCH --output=results/logs/validate_%x_%A_%a.out
#SBATCH --error=results/logs/validate_%x_%A_%a.out

GRAPHS=(anna jean myciel6 queen6_6 queen7_7 queen8_8 queen8_12 queen9_9 queen11_11 queen13_13 queen5_5)
G=${GRAPHS[$SLURM_ARRAY_TASK_ID]}
echo "host=$(hostname) graph=$G"
/home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python -u scripts/validate_step_a.py --graphs $G --seeds 5 --embed-init normal --save-dir results/trajectories
