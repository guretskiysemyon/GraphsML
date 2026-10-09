#!/bin/bash
# Run pytest on the given test files (default: the whole tests/ dir) as a Slurm job, CPU only.
# Usage: sbatch scripts/run_tests.sbatch tests/test_controls.py tests/test_endpoint.py
#SBATCH --job-name=pytest_graphsml
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-00:20:00
#SBATCH --exclude=s-002
#SBATCH --requeue
#SBATCH --chdir=/home/morg/NLP_2526b/semyong/GraphsML
#SBATCH --output=results/logs/pytest_%j.out
#SBATCH --error=results/logs/pytest_%j.out

echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date +%T) args=$*"
/home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python -m pytest -v "${@:-tests}"
echo "pytest exit code $? end=$(date +%T)"
