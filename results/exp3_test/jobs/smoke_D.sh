#!/bin/bash
set -e
PY=/home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/python
$PY -u scripts/tune_linearised.py --graphs planted_n60_k5 --n-random 2 --n-epochs 300 --eval-every 100 --seeds 1 --device cpu --out results/exp3_test/D_tune_stage1
$PY -u scripts/summarize_tuning.py --dir results/exp3_test/D_tune_stage1
$PY -u scripts/tune_linearised.py --write-stage2 results/exp3_test/D_tune_stage1
$PY -u scripts/run_linearised.py --device cpu --graph queen5_5 --seeds 0 1 --config results/exp3_test/linearised_smoke_config.json --t-max 3000 --max-steps-factor 2 --out results/exp3_test
$PY -u scripts/run_movement.py --graph queen5_5 --seeds 0 1 --runs results/exp3_test/D
echo "smoke D exit 0"
