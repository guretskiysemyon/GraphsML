# Experiment 3 (RQ2) — progress log

Last updated: 2026-10-11.

Plan, design choices, run order and the draft interpretation rules: [Experiment3_Steps.md](Experiment3_Steps.md). This file records what was done and the run log.

## 0. State at 2026-10-11

- Arm A (PI-GNN) = the Experiment 1 sweep, finished; nothing to rerun.
- Free-logit arms B / B-rand, tuning, wall-clock matching, 4C/4E diagnostics, analysis: **implemented and tested**, not yet run on DIMACS.
- **Full pipeline submitted by Semyon on 2026-10-11 as a Slurm dependency chain** (run log #4–#15). Its choices: A without dropout as the noise control, the §4.7 search space, the stage-2 winner frozen automatically by `scripts/freeze_free_logits.py`. The analysis (#15) runs automatically at the end, so the §6 draft interpretation rules have to be confirmed **before** opening `results/analysis/exp3_*`.

## 1. Done

- [x] Code for rungs B and B-rand, tuning, main runs, diagnostics, analysis (files listed in Experiment3_Steps.md §3). `scripts/run_sweep.py` gained `--config` (default unchanged, so Experiment 1 reruns behave as before).
- [x] Unit tests: 13 in `tests/test_free_logits.py` pass, existing training/model/loss tests still pass (run log #1, #2).
- [x] End-to-end smoke test on queen5_5 with an untuned placeholder config (Adam, lr 0.1), output in `results/exp3_test/` (not for analysis), run log #3. Timing: ~1.7 ms per free-logit step against ~6.9 ms per PI-GNN epoch (both 5 processes per GPU), so matching A's 100k-epoch wall time needs ~400k free-logit steps.

## 2. Run log (every script through `sbatch`)

| # | script | what | job id | node / time | status |
| --- | --- | --- | --- | --- | --- |
| 1 | `run_tests.sbatch tests/test_free_logits.py tests/test_train_checkpointed.py tests/test_model.py tests/test_losses.py` | new free-logit tests + existing training tests | 1011572 | 2026-10-11 ~18:41–18:45 | 24 passed, 3 failed: a **test** bug. On a 10-cycle with k = 2 all three optimisers stop at 2 conflicts, two "domain walls" where every single flip is sideways, i.e. a genuine single-flip-optimal local minimum. Test changed to a star graph, where every single-flip-optimal 2-colouring is proper |
| 2 | `run_tests.sbatch tests/test_free_logits.py` | rerun after the test fix | 1011575 | 2026-10-11 ~18:46–18:47 | **13 passed** |
| 3 | `results/exp3_test/jobs/smoke.sh` (`sbatch` with 1 GPU) | smoke test: tuning on 2 tuning graphs (5 configs, 300 steps) → summarize → stage-2 configs; B and B-rand on queen5_5, 5 seeds, step budget 5k, cap 200k; rerun of B seed 0 (must skip); `run_movement` on B and B-rand; `run_soft` on B seed 0; `analyze_exp3` → `results/exp3_test/` | 1011589 (1011576: first attempt, failed because the placeholder config was in node-local /tmp) | s-003, 2026-10-11 18:49–19:05 | done, exit 0, every stage ran. Time targets A@50k/A@100k missed **as expected** (cap 200k steps on purpose); rerun of B seed 0 skipped; `run_soft` entropy check 1.1e-7. Found: `analyze_exp3.py` treated non-arm entries of the exp3 folder as arms; fixed (only folders containing `<g>_seed<s>/DONE`) |
| 4 | `tune_free_stage1.sbatch` | tuning stage 1 | array 1011593 | 2026-10-11 from ~18:55 | submitted (Semyon) |
| 5 | `summarize_tuning.py` + `tune_free_logits.py --write-stage2` (`--wrap`) | stage-1 summary, stage-2 input | 1011605 (`afterok:1011593`) | | queued |
| 6 | `tune_free_stage2.sbatch` | tuning stage 2 | array 1011606 (`afterok:1011605`) | | queued |
| 7 | `summarize_tuning.py` + `freeze_free_logits.py` (`--wrap`) | stage-2 summary, write `configs/free_logits_frozen.json` | 1011607 (`afterok:1011606`) | | queued |
| 8 | `run_free_logits.sbatch` (ARM=B) | arm B, 68 graphs × 5 seeds | array 1011608 (`afterok:1011607`) | | queued |
| 9 | `run_free_logits.sbatch` (ARM=Brand) | arm B-rand | array 1011609 (`afterok:1011607`) | | queued |
| 10 | `run_sweep.sbatch` (CONFIG=pignn_frozen_nodropout, OUT=results/exp3/A_nodrop) | noise control, 100k | array 1011600 | | queued |
| 11 | `run_movement.sbatch` (arm A) | 4C/4E for A | array 1011599 | from ~19:00 | running (270/340 runs done at 19:10) |
| 12 | `run_movement.sbatch` (A_nodrop) | 4C/4E | array 1011601 (`afterany:1011600`) | | queued |
| 13 | `run_movement.sbatch` (B) | 4C/4E | array 1011610 (`afterany:1011608`) | | queued |
| 14 | `run_movement.sbatch` (Brand) | 4C/4E | array 1011611 (`afterany:1011609`) | | queued |
| 15 | `analyze_exp3.py` (`--wrap`) | tables + Figure 3 | 1011612 (`afterany:` #11–#14) | | queued |
| 4 | `tune_free_stage1.sbatch` | tuning stage 1: 43 configs × 13 graphs × 2 regimes × 2 seeds × 10k steps → `results/tuning/free_stage1/` | array 1011593 | 2026-10-11 | submitted |
| 5 | `summarize_tuning.py --dir results/tuning/free_stage1` + `tune_free_logits.py --write-stage2` (`sbatch --wrap`, CPU) | stage-1 ranking → `results/tuning/free_stage2_input_configs.json` | 1011605 (afterok #4) | | submitted |
| 6 | `tune_free_stage2.sbatch` | tuning stage 2: 10 configs × 3 seeds × 50k steps → `results/tuning/free_stage2/` | array 1011606 (afterok #5) | | submitted |
| 7 | `summarize_tuning.py --dir results/tuning/free_stage2` + `freeze_free_logits.py` (`sbatch --wrap`, CPU) | stage-2 winner (lowest mean rank, as for PI-GNN) → `configs/free_logits_frozen.json` | 1011607 (afterok #6) | | submitted |
| 8 | `run_free_logits.sbatch` (`ARM=B`) | arm B, 68 graphs × 5 seeds, step budget = A's, time-matched to A (cap 10×) → `results/exp3/B/` | array 1011608 (afterok #7) | | submitted |
| 9 | `run_free_logits.sbatch` (`ARM=Brand`) | arm B-rand, same → `results/exp3/Brand/` | array 1011609 (afterok #7) | | submitted |
| 10 | `run_movement.sbatch` | 4C/4E for arm A → new `results/main/*/movement.jsonl` (no existing file touched) | array 1011599 | 2026-10-11 | submitted |
| 11 | `run_sweep.sbatch` (`CONFIG=configs/pignn_frozen_nodropout.json OUT=results/exp3/A_nodrop`) | noise control: PI-GNN without dropout, 68 × 5 × 100k epochs | array 1011600 | 2026-10-11 | submitted |
| 12 | `run_movement.sbatch` (`RUNS=results/exp3/A_nodrop`) | 4C/4E for A-nodrop | array 1011601 (afterany #11) | | submitted |
| 13 | `run_movement.sbatch` (`RUNS=results/exp3/B`, `RUNS=results/exp3/Brand`) | 4C/4E for B and B-rand | arrays 1011610 (afterany #8), 1011611 (afterany #9) | | submitted |
| 14 | `analyze_exp3.py` (`sbatch --wrap`, CPU) | `results/analysis/exp3_*` | 1011612 (afterany #10, #12, #13) | | submitted, **held** until the §6 rules are confirmed |
