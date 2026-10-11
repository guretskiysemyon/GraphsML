#!/bin/bash
# Experiment 3 job chain after stage-1 tuning (job $1). Run once with bash.
set -e
cd /home/morg/NLP_2526b/semyong/GraphsML
PY=/home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python
S1=$1; MA=$2; MND=$3
C=(-A gpu-students -p studentkillable --exclude=s-002 --chdir=$PWD --parsable)
W=$(sbatch "${C[@]}" --dependency=afterok:$S1 --cpus-per-task=1 --mem=4G --time=0-00:30:00 -J free_s2cfg -o results/logs/free_stage2cfg_%j.out --wrap "$PY scripts/summarize_tuning.py --dir results/tuning/free_stage1 && $PY scripts/tune_free_logits.py --write-stage2 results/tuning/free_stage1"); echo W=$W
S2=$(sbatch --parsable --dependency=afterok:$W scripts/tune_free_stage2.sbatch); echo S2=$S2
F=$(sbatch "${C[@]}" --dependency=afterok:$S2 --cpus-per-task=1 --mem=4G --time=0-00:30:00 -J free_freeze -o results/logs/free_freeze_%j.out --wrap "$PY scripts/summarize_tuning.py --dir results/tuning/free_stage2 && $PY scripts/freeze_free_logits.py"); echo F=$F
B=$(sbatch --parsable --dependency=afterok:$F --export=ALL,ARM=B scripts/run_free_logits.sbatch); echo B=$B
BR=$(sbatch --parsable --dependency=afterok:$F --export=ALL,ARM=Brand scripts/run_free_logits.sbatch); echo BR=$BR
MB=$(sbatch --parsable --dependency=afterany:$B --export=ALL,RUNS=results/exp3/B scripts/run_movement.sbatch); echo MB=$MB
MBR=$(sbatch --parsable --dependency=afterany:$BR --export=ALL,RUNS=results/exp3/Brand scripts/run_movement.sbatch); echo MBR=$MBR
AN=$(sbatch "${C[@]}" --dependency=afterany:$MA:$MB:$MBR:$MND --cpus-per-task=2 --mem=8G --time=0-01:00:00 -J analyze_exp3 -o results/logs/analyze_exp3_%j.out --wrap "$PY scripts/analyze_exp3.py"); echo AN=$AN
