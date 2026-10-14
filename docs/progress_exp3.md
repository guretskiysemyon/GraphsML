# Experiment 3 (RQ2) — progress log

Last updated: 2026-10-14 (§6: end-to-end D-CE view from Experiment 2).

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
| 14 | `analyze_exp3.py` (`sbatch --wrap`, CPU) | `results/analysis/exp3_*` | 1011612 (afterany #10, #12, #13) | | held until the §6 rules were confirmed (2026-10-12), then released; done, exit 0, 46 s. Results in §3 |
| 15 | `analyze_exp3.py --exp3 results/exp3_test` (`sbatch --wrap`) | test of the extended analysis (both co-primary metrics, both gate rules, per-graph and per-family verdicts, after-descent panels) on the smoke-test data only | 1014413 | 2026-10-12 | done; worked (1 graph, no real results) |
| 16 | `analyze_exp3.py` (`sbatch --wrap`) | rerun, figure only: y axes log instead of symlog (the after-descent panels had lost their tick labels) | 1014430 | 2026-10-12 | done; `exp3_summary.csv` byte-identical to #14 |

## 3. Results (analysis #14/#16, 2026-10-12; rules of Experiment3_Steps §6, confirmed before the analysis)

Files: `results/analysis/exp3_summary.csv`, `exp3_readings.csv`, `exp3_per_graph.csv`, `exp3_verdicts_per_graph.csv`, `exp3_by_family.csv`, `exp3_figure3.png`. 68 graphs × 5 seeds per arm, k = χ − 1. Frozen free-logit config: AdamW, lr 0.0427, wd 0.064 (stage-2 winner, mean rank 2.38 vs runner-up 2.60). Every B / B-rand run reached all of A's time targets.

**Primary, at 100k (X = B, Y = A; win/tie/loss counted for B):**

| metric | view | B better | ties | A better | median d | p | reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| argmax conflicts | steps | 32 | 23 | 13 | −0.086 | 0.0018 | **B better** |
| argmax conflicts | time | 32 | 23 | 13 | −0.086 | 0.0018 | **B better** |
| after descent | steps | 17 | 25 | 26 | +0.060 | 0.0007 | **A better** |
| after descent | time | 17 | 26 | 25 | +0.060 | 0.0008 | **A better** |
| B vs B-rand (both metrics) | steps | 14 | 42 | 12 | −0.002 | 0.020 | not distinguishable |

**Gate:** argmax: closed under both rules (B is better). After descent: **open under both rules** (A better per step and per second; the views agree).

**What it means (descriptive):**
- **Free logits always end at a single-flip local minimum:** the single-flip-optimal fraction is 1.00 for B and B-rand on all 68 graphs (PI-GNN 0.42, PI-GNN without dropout 0.17), so descent never changes them and their argmax and after-descent numbers are identical. This is what the Potts lemma predicts for direct optimisation in P-space; PI-GNN, optimising θ, stops at saturated points that are often not single-flip optimal (RQ1, §1 of progress_exp1).
- So the two metrics answer different questions. **Raw optimiser quality (argmax):** free logits are better, mainly because PI-GNN's raw argmax is not yet a local minimum (mean 1517 conflicts vs 109, dominated by DSJR/DSJC). **Quality of the local minimum reached (after descent):** PI-GNN + descent ends lower, a small but consistent effect (median +6%, 26 vs 17 graphs). By family it is concentrated in queen (8 A / 1 B), le (5 / 3), and the register-allocation graphs fpsol, inithx, mulsol (10 / 0); free logits do better on flat (4 / 0), DSJC, school and miles.
- **Matched time ≈ matched steps:** free logits converge within ~10³ steps (Figure 3), so running them ~2.9× more steps to match A's wall time (median 288k steps at A@100k) changes nothing.
- **Starting point:** B vs B-rand is not distinguishable (42 ties), so PI-GNN's Z0 carries no useful information for free logits.
- **Dropout (noise control, secondary):** PI-GNN without dropout is worse than with dropout on both metrics (after descent: 21 A better, 40 ties, 7 worse), and free logits vs no-dropout PI-GNN after descent is not distinguishable (p = 0.70). So part of PI-GNN's after-descent advantage over free logits is tied to dropout noise; by the §6 table ("A_nodrop ≈ B while A ≠ B") this points to **dropout noise rather than message passing** as a driver of the A-vs-B after-descent difference. Secondary, no test claim.
- Final budget (300k for the 13 extended graphs): same picture (argmax 31/24/13 for B; after descent 14/28/26 for A).

**Next, per the rules:** the after-descent reading opens the mechanism gate (rungs C and D, 4A/4B/4D), but the noise control suggests checking dropout first: e.g. free logits with logit noise (`run_free_logits.py --logit-noise σ`), or rung C, to separate noise from message passing. User decision.

## 4. Follow-ups (user, 2026-10-12: "run this": items 1-3 of the follow-up list; rung D and the spectral analysis held)

Code added: rung C = `NodeMLP` in `src/graphsml/model.py` (PI-GNN with both layers' neighbour terms deleted: same widths, dropout, initialisation; about half PI-GNN's non-embedding weights), `build_model(arch, ...)`, `TrainConfig.arch` / `train_lean(arch=)` (default "sage" = PI-GNN, unchanged), `tune_stage1.py --arch mlp --write-stage2`, `tune.pignn_stage2_configs`, `freeze_free_logits.py --keys/--set`, `scripts/tune_C_stage{1,2}.sbatch`, `scripts/analyze_movement.py`; `analyze_exp3.py` now also compares every follow-up arm with B. Tests: `tests/test_mlp.py` (6), 33 passed in total (job 1014652).

| # | script | what | job | status |
| --- | --- | --- | --- | --- |
| 17 | `run_free_logits.sbatch` (`ARM=B`, `EXTRA="--logit-noise σ"`) | noise check: free logits (frozen B config) with Gaussian logit noise in the training forward pass, σ = 0.5 / 1 / 2, step- and time-matched to A → `results/exp3/B_noise{0.5,1,2}/` | arrays 1014610, 1014611, 1014612 | submitted |
| 18 | `run_tests.sbatch tests/test_mlp.py ...` | rung C tests + existing | 1014652 | 33 passed |
| 19 | `analyze_movement.py` (`sbatch --wrap`) | item 1: diagnostics of the four finished arms → `results/analysis/exp3_movement*` | 1014655 | submitted |
| 20 | `tune_C_stage1.sbatch` | rung C tuning stage 1 (PI-GNN's 43 stage-1 configs, `--arch mlp`) → `results/tuning/C_stage1/` | array 1014664 | submitted |
| 21 | `summarize_tuning` + `tune_stage1.py --write-stage2` | → `results/tuning/C_stage2_input_configs.json` | 1014665 (afterok 20) | submitted |
| 22 | `tune_C_stage2.sbatch` | stage 2, 10 configs × 3 seeds × 50k | array 1014666 (afterok 21) | submitted |
| 23 | `summarize_tuning` + `freeze_free_logits.py --keys lr hidden_dim embed_dim dropout --set {"arch":"mlp"}` | → `configs/mlp_frozen.json` | 1014667 (afterok 22) | submitted |
| 24 | `run_sweep.sbatch` (`CONFIG=configs/mlp_frozen.json OUT=results/exp3/C T_MAX=100000`) | rung C main run, 68 × 5 × 100k (matched steps only) | array 1014668 (afterok 23) | submitted |
| 25 | `run_movement.sbatch` | 4C/4E for C and the three noise arms | 1014669 (afterany 24), 1014670–1014672 (afterany 17) | submitted |
| 26 | `analyze_exp3.py && analyze_movement.py` | final rerun with all arms | 1014673 (afterany 25) | submitted |

Readings fixed in advance (descriptive, secondary arms, no tests beyond the reported Wilcoxon p): **noise** — if some σ makes B_noise ≈ A after descent (tie rule / median |d| ≤ 0.05) while B < A, PI-GNN's after-descent advantage is attributed to noise injection; if no σ closes the gap, noise alone does not explain it. **Rung C** — C ≈ A: the graph inside the network is not what matters (shared weights + dropout); C ≈ B: message passing matters; in between: both contribute.

**Item 1 result (job #19 rerun after a filter fix: time-matched checkpoints that coincide with grid checkpoints were dropped, leaving 55/64 graphs at step 1 for B/B-rand; now 68 everywhere). `results/analysis/exp3_movement*`, descriptive, medians over 68 graphs at 100k:**
- **Saturation comes from dropout, not from the GNN.** Median logit margin: PI-GNN 204, PI-GNN without dropout 18, free logits 18 (B and B-rand). PI-GNN's logits move ~88× their initial (row-centred) size, without dropout ~11×, free logits ~2×. So RQ1's "saturated endpoint with zero gradient on improvable vertices" is a dropout effect: the network makes its outputs extremely confident to survive the noise.
- **The GNN commits much earlier.** Vertices with p_max ≥ 0.99: at step 100, PI-GNN 69 %, without dropout 89 %, free logits 0 % (B) / 9 % (B-rand); free logits reach 55 % (B) only at step 1,000. Free logits lag in argmax conflicts for the first ~100 steps, then drop below PI-GNN by step ~200 and stay there.
- **All arms end (almost) hard:** Dirichlet energy / 2m ≈ 0.99 and softness gap ≤ 10⁻³; free logits are fully one-hot (softness gap ~10⁻⁵, p_max ≥ 0.99 on 100 % of vertices) and single-flip optimal; PI-GNN keeps a small residual softness (DSJR/DSJC).
- Relative parameter movement (free logits 3.1×, PI-GNN 1.4×) is not comparable across parameterisations; only within an arm.

## 5. Rung D (user, 2026-10-12: "run D as well after this")

**What it is.** `src/graphsml/linearised.py`: PI-GNN linearised at its initialisation, Z(δ) = Z0 + J0 δ, with δ = θ − θ0 trained on the Potts loss (J0 = PI-GNN's eval-mode Jacobian at θ0, never materialised: one forward-mode JVP per step, gradients by reverse mode through it). With plain gradient descent this is exactly the plan's rung D update Z ← Z − η K0 ∇_Z L with K0 = J0 J0ᵀ (checked against an explicit Jacobian in `tests/test_linearised.py`, 5 passed), i.e. PI-GNN's own graph-structured preconditioner frozen at init; with Adam it is the plan's J D_t Jᵀ variant. Same Z0 as A and B (checked against `z0.pt` per run). Differences from A to keep in mind: no dropout, no weight decay, J frozen. Tuning: same protocol (3 references: Adam at PI-GNN's lr, SGD 0.01, SGD 0.1; 40 draws, optimiser ∈ {Adam, SGD}; stage 2 top 6 + 4). Main run: 100k-step budget (one step ≈ 2–3 PI-GNN steps), time-matched to A, 4 h limit. Smoke test on CPU passed (1014747; Z0 agrees with A's to 2e-6).

| # | script | what | job | status |
| --- | --- | --- | --- | --- |
| 27 | `run_tests.sbatch tests/test_linearised.py` | rung D tests | (see log) | 5 passed |
| 28 | `results/exp3_test/jobs/smoke_D.sh` (CPU) | tuning/summary/stage-2 writer, run_linearised on queen5_5 (2 seeds, 3k steps), run_movement | 1014747 (GPU attempt 1014697 cancelled: GPU quota full) | passed |
| 29 | `tune_D_stage1.sbatch` → `D_s2cfg` → `tune_D_stage2.sbatch` → `D_freeze` | tuning → `configs/linearised_frozen.json` | 1014778 → 1014779 → 1014780 → 1014781 | submitted |
| 30 | `run_linearised.sbatch` → `run_movement.sbatch RUNS=results/exp3/D` | main run, 68 × 5 → `results/exp3/D/` | 1014782 → 1014783 | submitted |
| 31 | `analyze_exp3.py && analyze_movement.py` | final analysis with every arm (A, A_nodrop, B, Brand, B_noise*, C, D) | 1014784 (afterany 1014783 and the earlier final 1014673) | submitted |

Readings fixed in advance (plan's table; descriptive): **D ≈ A while B differs** → graph-structured coupling (fixed) explains much of the GNN effect; **A better than both B and D** → the full, evolving nonlinear GNN dynamics contribute; **D ≈ B** → a fixed graph preconditioner adds nothing over free logits. Because D has no dropout, D vs A_nodrop is reported as well (`analyze_exp3.py` now compares every follow-up arm with A, B and A_nodrop).

**2026-10-13, C freeze failure and fix.** `C_freeze` (1014667) failed: `freeze_free_logits.py` called `Path.relative_to(ROOT)` on the relative paths passed on the command line (ValueError); the C sweep 1014668 was left `DependencyNeverSatisfied`. Fixed (`.resolve().relative_to(ROOT)`, before `D_freeze` 1014781 runs, so D is not affected). Cancelled 1014668, 1014669 and the intermediate analysis 1014673; resubmitted C freeze 1018985 (done: `configs/mlp_frozen.json` = lr 0.0281, hidden 39, embed 55, dropout 0.202; stage-2 mean rank 2.17 vs runner-up 2.50) → C sweep 1018986 → C movement 1018987; the final analysis 1014784 now waits for 1014783 (D movement) and 1018987 (C movement). Noise arms: 340/340 runs each for σ = 0.5, 1, 2, movement done.

### 2026-10-13 status and fix
- Noise check done: B_noise0.5/1/2, 340/340 runs each, movement done (jobs 1014610–1014612, 1014670–1014672).
- C tuning stages 1 and 2 done (1014664, 1014666). **C freeze 1014667 FAILED**: `freeze_free_logits.py` called `Path.relative_to(ROOT)` on the relative paths it was given. Fixed (`.resolve()` first; this also covers D's freeze, which had not run yet). The dead C chain (sweep 1014668, movement 1014669, intermediate analysis 1014673) was cancelled and resubmitted: freeze 1018985 (done → `configs/mlp_frozen.json`: lr 0.0281, hidden 39, embed 55, dropout 0.202, stage-2 mean rank 2.17 vs 2.50), sweep 1018986, movement 1018987. Final analysis 1014784 re-pointed to `afterany:1014783 (D movement), 1018987 (C movement)`.
- D tuning stage 1 done (1014778), stage 2 running (1014780).
- Interim analysis with the noise arms: 1018991.

**2026-10-13, analysis fix.** An analysis run at 12:01 (1018991, not from this log) picked up the 5 C graphs finished so far, and `analyze_exp3.py` restricted every comparison to the graphs all arms share, so all `exp3_*` outputs were overwritten with a 5-graph version. Fixed: the graph set is now A's 68, an incomplete arm only lowers the `graphs` count of its own comparisons (with a warning), and `--exclude <arm>` skips arms. Rerun without C and D: job 1019054; the A/B/B-rand/A_nodrop numbers are identical to §3.

### Noise check result (job 1019054, 68 graphs, 100k steps; descriptive, rules of §4)

Free logits with Gaussian logit noise in the training forward pass (eval on clean Z). Every noisy run still ends single-flip optimal (fraction 1.00), so argmax = after descent for all B arms.

| arm | vs PI-GNN (A), after descent: better / tie / worse, median d, p | vs plain free logits (B) | median conflicts (A: 11.8) |
| --- | --- | --- | --- |
| σ = 0.5 | 20 / 19 / 29, +0.058, p = 0.002 (A better) | 10 / 55 / 3, not distinguishable | 13.2 |
| σ = 1 | **35 / 30 / 3, −0.159, p < 1e-4 (noisy B better)** | 49 / 19 / 0 better | 7.7 |
| σ = 2 | **44 / 21 / 3, −0.275, p < 1e-4 (noisy B better)** | 53 / 15 / 0 better | 6.7 |

Same picture at matched time (σ = 2: 44 / 21 / 3) and against PI-GNN without dropout (σ = 2: 45 / 21 / 2).

**Reading (§4 rule):** noise does not just close the after-descent gap, it reverses it. With σ ≥ 1, free logits beat PI-GNN on most graphs and lose on 3; σ = 0.5 is too weak to matter. So PI-GNN's small after-descent advantage over plain free logits is consistent with being a noise-injection effect, and noise injected directly into free logits is more effective than PI-GNN's dropout. Diagnostics: noise keeps free logits softer (median margin 18 → 13 → 8 for σ = 0 / 1 / 2; p_max ≥ 0.99 on 93 % of vertices at σ = 2), unlike dropout, which drives PI-GNN to margins ~200.

**Caveats:** σ was not tuned on the synthetic set; three values were fixed in advance and all are reported, but σ = 2 being the best of three on DIMACS is a test-set observation, and the trend has not levelled off (larger σ untested). A fair "tuned noisy free logits" arm would tune σ on the tuning set. PI-GNN's dropout rate was tuned.

## 6. End-to-end D-CE view (from Experiment 2, 2026-10-14)

The plan asks for argmax as the measure of raw optimiser quality and D-CE as "an end-to-end secondary view". Experiment 2 ([Experiment2_Steps.md](Experiment2_Steps.md) §6, jobs 1037155–1037162 and analysis 1039926) decoded every arm's P at step 100k with argmax and with D-CE (saturation order), each followed by the same strict descent. Descriptive, Experiment 1's tie rule (|d| ≤ 5 % or < 1 conflict), 68 graphs.

| arm | mean after argmax + descent | mean after D-CE + descent | rank (argmax → D-CE) | vs A after D-CE (better / tie / worse) |
| --- | --- | --- | --- | --- |
| B_noise2 | 83.4 | 83.4 | 1 → 1 | 35 / 29 / 4 |
| B_noise1 | 92.5 | 92.3 | 2 → 2 | 31 / 30 / 7 |
| Brand | 107.1 | 106.2 | 3 → 3 | 14 / 25 / 29 |
| B | 108.6 | 107.3 | 4 → 4 | 14 / 24 / 30 |
| B_noise0.5 | 109.2 | 108.4 | 5 → 5 | 13 / 24 / 31 |
| A (PI-GNN) | 114.9 | 111.4 | 6 → 6 | — |
| A_nodrop | 115.7 | 117.0 | 7 → 8 | 4 / 34 / 30 |
| C | 120.4 | 116.1 | 8 → 7 | 2 / 44 / 22 |
| D | 130.4 | 123.1 | 9 → 9 | 1 / 21 / 46 |

(Means are over graphs and are pulled by a few large graphs; the paired counts are the primary view.)

**Reading.** The §3/§4 conclusions do not depend on the decoder: the arms keep their order under D-CE and the per-graph spread across arms shrinks only modestly (median (max − min)/mean 0.87 → 0.68). D-CE ties argmax + descent on 63–68 graphs for every free-logit arm and helps the GNN-type arms a little (A 22, C 27, D 45 graphs better), and PI-GNN's small per-graph edge over plain B persists under D-CE (B better / tie / worse than A: 17 / 25 / 26 after argmax, 14 / 24 / 30 after D-CE), even though B has the lower mean (the means are pulled by a few large graphs). Noisy free logits stay ahead of PI-GNN; σ = 2 is the only arm whose D-CE result beats the no-information control (D-CE on uniform P = DSATUR), 36 / 27 / 5. The untuned-σ caveat of the noise check applies unchanged.
