#!/bin/bash
# Experiment 3 follow-ups (2026-10-12): rung C chain + movement for the noise arms + final analyses.
# Args: the three noise-check array ids.
set -e
cd /home/morg/NLP_2526b/semyong/GraphsML
PY=/home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python
N1=$1; N2=$2; N3=$3
C=(-A gpu-students -p studentkillable --exclude=s-002 --chdir=$PWD --parsable)
CS1=$(sbatch --parsable scripts/tune_C_stage1.sbatch); echo CS1=$CS1
CW=$(sbatch "${C[@]}" --dependency=afterok:$CS1 --cpus-per-task=1 --mem=4G --time=0-00:30:00 -J C_s2cfg -o results/logs/C_stage2cfg_%j.out --wrap "$PY scripts/summarize_tuning.py --dir results/tuning/C_stage1 && $PY scripts/tune_stage1.py --write-stage2 results/tuning/C_stage1"); echo CW=$CW
CS2=$(sbatch --parsable --dependency=afterok:$CW scripts/tune_C_stage2.sbatch); echo CS2=$CS2
CF=$(sbatch "${C[@]}" --dependency=afterok:$CS2 --cpus-per-task=1 --mem=4G --time=0-00:30:00 -J C_freeze -o results/logs/C_freeze_%j.out --wrap "$PY scripts/summarize_tuning.py --dir results/tuning/C_stage2 && $PY scripts/freeze_free_logits.py --stage1 results/tuning/C_stage1_summary.csv --stage2 results/tuning/C_stage2_summary.csv --stage2-input results/tuning/C_stage2_input_configs.json --out configs/mlp_frozen.json --keys lr hidden_dim embed_dim dropout --set '{\"arch\": \"mlp\"}' --tuning-start \"the model's own random initialisation (seeded as PI-GNN)\""); echo CF=$CF
CR=$(sbatch --parsable --dependency=afterok:$CF --export=ALL,CONFIG=configs/mlp_frozen.json,OUT=results/exp3/C,T_MAX=100000 scripts/run_sweep.sbatch); echo CR=$CR
CM=$(sbatch --parsable --dependency=afterany:$CR --export=ALL,RUNS=results/exp3/C scripts/run_movement.sbatch); echo CM=$CM
M1=$(sbatch --parsable --dependency=afterany:$N1 --export=ALL,RUNS=results/exp3/B_noise0.5 scripts/run_movement.sbatch); echo M1=$M1
M2=$(sbatch --parsable --dependency=afterany:$N2 --export=ALL,RUNS=results/exp3/B_noise1 scripts/run_movement.sbatch); echo M2=$M2
M3=$(sbatch --parsable --dependency=afterany:$N3 --export=ALL,RUNS=results/exp3/B_noise2 scripts/run_movement.sbatch); echo M3=$M3
AN=$(sbatch "${C[@]}" --dependency=afterany:$CM:$M1:$M2:$M3 --cpus-per-task=2 --mem=8G --time=0-01:00:00 -J analyze_exp3 -o results/logs/analyze_exp3_%j.out --wrap "$PY scripts/analyze_exp3.py && $PY scripts/analyze_movement.py"); echo AN=$AN
