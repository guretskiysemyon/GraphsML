#!/bin/bash
set -e
PY=/home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python
echo "host=$(hostname) start=$(date +%T)"
$PY -u scripts/tune_free_logits.py --graphs planted_n60_k5 regular_n200_d20 --n-random 2 --n-epochs 300 --eval-every 100 --seeds 1 --device cuda --out results/exp3_test/tune_stage1
$PY -u scripts/summarize_tuning.py --dir results/exp3_test/tune_stage1
$PY -u scripts/tune_free_logits.py --write-stage2 results/exp3_test/tune_stage1
$PY -u scripts/run_free_logits.py --graph queen5_5 --arm B --seeds 0 1 2 3 4 --config results/exp3_test/free_smoke_config.json --t-max 5000 --max-steps-factor 40 --out results/exp3_test
$PY -u scripts/run_free_logits.py --graph queen5_5 --arm Brand --seeds 0 1 2 3 4 --config results/exp3_test/free_smoke_config.json --t-max 5000 --max-steps-factor 40 --out results/exp3_test
$PY -u scripts/run_free_logits.py --graph queen5_5 --arm B --seeds 0 --config results/exp3_test/free_smoke_config.json --t-max 5000 --max-steps-factor 40 --out results/exp3_test
$PY -u scripts/run_movement.py --graph queen5_5 --runs results/exp3_test/B
$PY -u scripts/run_movement.py --graph queen5_5 --runs results/exp3_test/Brand
$PY -u scripts/run_soft.py --graph queen5_5 --seeds 0 --main results/exp3_test/B
$PY -u scripts/analyze_exp3.py --exp3 results/exp3_test --out results/exp3_test/analysis
echo "smoke exit 0 end=$(date +%T)"
