#!/bin/bash
# Experiment 3 rung D chain (2026-10-12). Arg: the previous final-analysis job (C + noise arms).
set -e
cd /home/morg/NLP_2526b/semyong/GraphsML
PY=/home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python
PREV=$1
C=(-A gpu-students -p studentkillable --exclude=s-002 --chdir=$PWD --parsable)
DS1=$(sbatch --parsable scripts/tune_D_stage1.sbatch); echo DS1=$DS1
DW=$(sbatch "${C[@]}" --dependency=afterok:$DS1 --cpus-per-task=1 --mem=4G --time=0-00:30:00 -J D_s2cfg -o results/logs/D_stage2cfg_%j.out --wrap "$PY scripts/summarize_tuning.py --dir results/tuning/D_stage1 && $PY scripts/tune_linearised.py --write-stage2 results/tuning/D_stage1"); echo DW=$DW
DS2=$(sbatch --parsable --dependency=afterok:$DW scripts/tune_D_stage2.sbatch); echo DS2=$DS2
DF=$(sbatch "${C[@]}" --dependency=afterok:$DS2 --cpus-per-task=1 --mem=4G --time=0-00:30:00 -J D_freeze -o results/logs/D_freeze_%j.out --wrap "$PY scripts/summarize_tuning.py --dir results/tuning/D_stage2 && $PY scripts/freeze_free_logits.py --stage1 results/tuning/D_stage1_summary.csv --stage2 results/tuning/D_stage2_summary.csv --stage2-input results/tuning/D_stage2_input_configs.json --out configs/linearised_frozen.json --keys lr optimizer weight_decay --set '{\"rung\": \"D (PI-GNN linearised at init)\"}' --tuning-start \"PI-GNN's frozen-config network at its seeded initialisation, linearised (Z0 = A's)\""); echo DF=$DF
DR=$(sbatch --parsable --dependency=afterok:$DF scripts/run_linearised.sbatch); echo DR=$DR
DM=$(sbatch --parsable --dependency=afterany:$DR --export=ALL,RUNS=results/exp3/D scripts/run_movement.sbatch); echo DM=$DM
AN=$(sbatch "${C[@]}" --dependency=afterany:$DM:$PREV --cpus-per-task=2 --mem=8G --time=0-01:00:00 -J analyze_exp3 -o results/logs/analyze_exp3_%j.out --wrap "$PY scripts/analyze_exp3.py && $PY scripts/analyze_movement.py"); echo AN=$AN
