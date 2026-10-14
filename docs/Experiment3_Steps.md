# Experiment 3 — Does the GNN parameterisation matter? (RQ2)

**RQ2.** Does the GNN parameterisation reach better minima than directly optimising one logit vector per vertex? If it does, can a fixed graph-dependent preconditioner reproduce the difference?

In `PI_GNN_Mechanistic_Plan_Final_v3.pdf` this is **Experiment 3** (§5, rungs A–D; §6 diagnostics 4A–4E). The plan's Experiment 2 answers RQ3 (decoders and null-P controls) and is a separate piece of work: [Experiment2_Steps.md](Experiment2_Steps.md), done 2026-10-14. It also ran every arm here through D-CE, the plan's "end-to-end secondary view" for this experiment (summary in [progress_exp3.md](progress_exp3.md) §6).

Started 2026-10-11. Same principles as Experiment 1: self-contained code, k = χ − 1 primary, graph = statistical unit, 5 seeds, every script through `sbatch`, interpretation rules written before results are seen.

---

## 1. Arms

| arm | parameterisation | status | where |
| --- | --- | --- | --- |
| **A** | PI-GNN, P = softmax(f_θ(G, X)) | **done** (Experiment 1 sweep: 68 graphs × 5 seeds, 100k epochs; 13 graphs to 300k) | `results/main/` |
| **B** | free logits, P = softmax(Z), Z ∈ ℝ^(n×k), **Z starts from A's exact Z₀** (same graph, same seed) | implemented, to run | `results/exp3/B/` |
| **B-rand** | free logits from random Gaussian logits with Z₀'s scale | implemented, to run | `results/exp3/Brand/` |
| noise control | **A without dropout** (recommended), or B with Gaussian logit noise | implemented (both options), to run | `results/exp3/A_nodrop/` or `results/exp3/B_noise<σ>/` |
| C (secondary, gated) | learned node embeddings → MLP, no message passing | not implemented | — |
| D (gated) | fixed graph-preconditioned descent Z ← Z − η M_G ∇_Z L | not implemented (needs M_G from 4A) | — |

The mandatory comparison is A vs B (dynamics) and B vs B-rand (starting point), at matched optimisation steps **and** matched wall-clock time. C, D and the 4A/4B/4D diagnostics run only if A differs from B (plan §6, §13 "mechanism gate").

---

## 2. What is reused from Experiment 1

Arm A does not need to be rerun. Every A run in `results/main/<graph>_seed<s>/` already saved everything Experiment 3 needs (progress_exp1.md decision 4):

- `z0.pt`: A's initial eval-mode logits Z₀, which B starts from (checked by a unit test: rebuilding Z₀ from the frozen config and seed reproduces the saved file bit-for-bit);
- `init_weights.pt` and `weights` in every checkpoint, for 4E's parameter distance;
- `Z` at every checkpoint, for P, Dirichlet energy and logit distance;
- `trajectory.jsonl` → `cumulative_seconds` per checkpoint: the wall-clock targets for the matched-time view;
- `endpoint_trajectory.jsonl`: argmax conflicts, single-flip report and descent at every checkpoint.

Also reused unchanged: the endpoint evaluator, the log checkpoint grid, the manifest (68 graphs, k = χ − 1), `summarize_tuning.py`, `run_soft.py` (g̃/H̃ works on free-logit run directories as they are).

---

## 3. What was implemented (2026-10-11)

| file | what |
| --- | --- |
| `src/graphsml/free_logits.py` | `FreeLogitConfig` (Adam / AdamW / SGD, weight decay, optional logit noise); `pignn_initial_logits` (rebuilds Z₀ for tuning graphs); `matched_random_logits` (B-rand start); `train_free_logits_lean` (tuning); `train_free_logits_checkpointed` (main runs: same files and resume semantics as `train_pignn_checkpointed`, plus wall-clock-matched checkpoints) |
| `src/graphsml/diagnostics.py` | Dirichlet energy (4C), logit / parameter distance from initialisation (4E), shared by all arms |
| `src/graphsml/tune.py` (appended) | free-logit search space, stage-1 sampler, stage-2 config builder |
| `scripts/tune_free_logits.py`, `tune_free_stage1.sbatch`, `tune_free_stage2.sbatch` | tuning, same protocol and effort as PI-GNN |
| `scripts/run_free_logits.py`, `run_free_logits.sbatch` | main DIMACS runs of B / B-rand (/ B + noise) |
| `scripts/run_movement.py`, `run_movement.sbatch` | always-logged 4C/4E diagnostics from saved checkpoints, for any arm (A included) |
| `scripts/analyze_exp3.py` | tables + Figure 3 |
| `scripts/run_sweep.py` (+ `.sbatch`) | new `--config` / `CONFIG`, `OUT` options, for the no-dropout PI-GNN control |
| `configs/pignn_frozen_nodropout.json` | frozen PI-GNN config with dropout 0 (not retuned) |
| `tests/test_free_logits.py` | 13 tests, passing (job 1011575) |

---

## 4. Design choices

1. **One step = one PI-GNN epoch** = one full-batch gradient step on the same `potts_loss` (dense, float32), on the same device type (GPU), with `loss.item()` every step as in A, and 5 seeds as 5 processes on one GPU as in A. These per-step costs are what the matched-time view compares.
2. **B starts from A's saved `z0.pt`** for the same graph and seed (fairness rule "start from the exact same initial logits").
3. **B-rand's start:** i.i.d. N(0, 1), row-centred, rescaled so its entry-wise std equals that of Z₀'s *row-centred* logits (softmax ignores per-row shifts, so that is the scale that matters). RNG `default_rng([seed, 5])`.
4. **B-rand reuses B's tuned config**, so B vs B-rand differs only in the start.
5. **Matched time.** For each A checkpoint e, the target is A's `cumulative_seconds` at e for the same graph and seed. B saves an extra checkpoint at the first step whose own cumulative seconds reach it (labelled `"A@e"` in `time_matched`) and keeps training past the step budget until every target is reached, capped at 10 × the step budget. Measured in the smoke test (queen5_5, 5 processes per GPU): **~1.7 ms per free-logit step vs ~6.9 ms per PI-GNN epoch**. So A's 100k epochs correspond to ~400k free-logit steps, within the 1M cap (3M cap for the 300k graphs). Recording extra checkpoints does not change the trajectory (unit-tested).
6. **Matched steps.** B's step budget = A's budget for that run (100k, or 300k for the 13 extended graphs), on a checkpoint grid that contains A's grid, so every A checkpoint has a B checkpoint at the same step.
7. **Tuning, same protocol and effort as PI-GNN** (progress_exp1.md decision 7). Stage 1: 3 references + 40 random draws × 13 tuning graphs × 2 regimes × 2 seeds × 10k steps. Stage 2: top 6 plus 4 draws from the top 6's log-space bounding box, × 3 seeds × 50k steps. Score: mean rank of final eval conflicts. Search space: optimiser ∈ {Adam, AdamW}, lr log-uniform in [1e-3, 1], AdamW weight decay log-uniform in [1e-4, 0.1]. References: PI-GNN's own optimiser and lr (AdamW, 0.0196, wd 0.01), Adam 0.01, Adam 0.1. Every tuning unit starts from PI-GNN's Z₀ for that graph, k and seed (B's start on DIMACS). No logit noise during tuning: B is the pure free-logit arm.
8. **Noise control.** PI-GNN trains with dropout 0.47; free logits have no noise. Recommended control: rerun A with dropout 0 (`configs/pignn_frozen_nodropout.json`, everything else frozen, same Z₀ and initial weights; costs one more GPU sweep, ~2 h as an array). The alternative, B + Gaussian logit noise (`--logit-noise σ`), needs a choice of σ, which would be one more tuned hyperparameter.
9. **Always-logged diagnostics** (4C Dirichlet energy, 4E logit and parameter distance) are computed after the fact from the saved checkpoints, in float64, by the same function for every arm (`run_movement.py`). For free logits the parameters are Z, so parameter distance = logit distance. For one-hot P, Dirichlet energy = 2(m − conflicts), so it is informative only while P is soft.
10. **Caveats to state in the report.** Wall-clock matched is not compute matched. A's timings were measured 2026-10-08/09 on whichever nodes ran the sweep, and B's will come from other jobs (host and GPU are recorded in `run_info.jsonl`; the cluster GPUs are TITAN Xp). A's per-epoch time includes dropout and two SAGE layers; B's includes only softmax + loss + Adam on one tensor.

---

## 5. Run order

Every step goes through `sbatch`; add a run-log row in `progress_exp3.md` first.

1. **Tests + smoke test.** Done (run log #1–#3).
2. **Tune stage 1** (GPU, 24 tasks; estimate from the smoke timings: 2,236 units × ~20 s ≈ 12 GPU-h ≈ 30–60 min wall):
   `sbatch scripts/tune_free_stage1.sbatch`
   then `python scripts/summarize_tuning.py --dir results/tuning/free_stage1` and `python scripts/tune_free_logits.py --write-stage2 results/tuning/free_stage1` (both through `sbatch --wrap`).
3. **Tune stage 2** (780 units × 50k steps ≈ 22 GPU-h ≈ 1–2 h wall): `sbatch scripts/tune_free_stage2.sbatch`, summarise with `--dir results/tuning/free_stage2`.
4. **Freeze** the winner as `configs/free_logits_frozen.json` with a `_provenance` block, as for `pignn_frozen.json`: keys `lr`, `optimizer`, `weight_decay`.
5. **Pilot** B on a few graphs to confirm the s/step estimate and that every time target is reached before the cap, including the slowest A runs: e.g. queen5_5, le450_15a and flat1000_60_0 (300k): `sbatch --array=2,47,67 --export=ALL,ARM=B scripts/run_free_logits.sbatch`. Check `DONE` → `missed_targets` is empty.
6. **Write and freeze the interpretation rules** (§6 below, currently a draft) before looking at any main result.
7. **Main runs:** `sbatch --export=ALL,ARM=B scripts/run_free_logits.sbatch` and `sbatch --export=ALL,ARM=Brand scripts/run_free_logits.sbatch` (68 tasks each).
8. **Noise control:** `sbatch --export=ALL,CONFIG=configs/pignn_frozen_nodropout.json,OUT=results/exp3/A_nodrop scripts/run_sweep.sbatch` (100k for all 68 graphs).
9. **Diagnostics:** `sbatch scripts/run_movement.sbatch` (A), `sbatch --export=ALL,RUNS=results/exp3/B scripts/run_movement.sbatch`, the same for `Brand` and `A_nodrop`. Optional: `run_soft.py --main results/exp3/B` for g̃/H̃.
10. **Analysis:** `scripts/analyze_exp3.py` → `results/analysis/exp3_*.csv`, `exp3_figure3.png`.
11. **Gate.** If A differs from B (rule in §6), plan rungs C, D and diagnostics 4A, 4B, 4D (§8). Otherwise stop: the null result is the finding, and 4C/4E explain it.

---

## 6. Interpretation rules — CONFIRMED by the user 2026-10-12, before any comparison was computed

Fixed in advance, as in progress_exp1.md §9. The runs had finished, but no A-vs-B number had been computed or looked at (only completeness checks: DONE counts, stop reasons, missed time targets).

**User decisions (2026-10-12):** keep the draft wherever it was recommended (three primary comparisons, the Experiment 1 tie rule, 100k as the main budget), with two changes, "since we are experimenting, see as much information as possible":
- **Metric:** argmax conflicts **and** conflicts after strict vertex descent are **co-primary**. Each gets its own readings; the three primary comparisons use α = 0.05 / 3 within each metric (raw p-values are reported, so a stricter correction over both metrics can be applied when reading).
- **Gate when the views disagree:** both rules are computed and reported, the per-step rule below (which decides) and the alternative per-time rule ("the gate opens if A is better at matched time"). `results/analysis/exp3_readings.csv` has one column per rule.

**Per graph.** Mean over the 5 seeds of raw **argmax conflicts** (co-primary; the plan says "report argmax first as the measure of raw optimiser quality") and of **conflicts after strict vertex descent** (co-primary). `d = (X − Y) / max(Y, 1)`; **tie** if |X − Y| < 1 or |d| ≤ 0.05; otherwise X better / Y better.

**Primary comparisons**, for each co-primary metric (68 graphs; two-sided Wilcoxon signed-rank on d; α = 0.05 / 3 ≈ 0.017, Bonferroni over the three):
1. B vs A, matched steps, at 100k (the budget all 68 graphs share);
2. B vs A, matched time, B at the checkpoint labelled `A@100000`;
3. B vs B-rand, matched steps, at 100k.

**Secondary (descriptive, no claims):** the same comparisons at each graph's final A budget (300k for the 13 extended graphs); single-flip-optimal fraction; Potts loss; the whole trajectory (Figure 3); per family; A_nodrop vs A and vs B; Dirichlet energy and movement traces.

**Readings** (the plan's table, made operational; "A better" means median d(B vs A) > +0.05, p < α and A better on more graphs than B; "B better" is the mirror image; otherwise "not distinguishable"):

| outcome | reading |
| --- | --- |
| B ≈ A in both views | The GNN adds little beyond reparameterising the relaxed variables. Mechanism gate stays closed. |
| B better than A | Message passing may actively hurt (consistent with colouring being heterophilous). |
| A better than B (in either view) | The GNN changes the optimisation enough to justify the gated mechanistic follow-up (C, D, 4A, 4B, 4D). |
| views disagree (e.g. A better per step, B better per second) | Report both. The gate opens only on a per-step difference, since a per-second advantage of B alone is a cost statement, not a mechanism. |
| B-rand differs from B | The starting point, not the dynamics, drives part of the A-vs-B difference. |
| A_nodrop ≈ B while A ≠ B | The A-vs-B difference comes from dropout noise, not message passing. |

---

## 7. Decisions for the user (before step 2 / step 6)

1. ~~Noise control~~ **Done:** A without dropout (run 2026-10-11).
2. ~~Search space~~ **Done** without SGD (tuned 2026-10-11/12). Was: free-logit search space as in §4.7 (Adam/AdamW, lr in [1e-3, 1], wd in [1e-4, 0.1]). Should plain SGD be included? The plan's 4B wants plain gradient descent later anyway, but in the tuning space it would split the budget across very different lr scales.
3. ~~The draft rules in §6~~ **Decided 2026-10-12** (see §6): argmax and after-descent co-primary; both gate rules reported, per-step decides.
4. Whether rung C (embedding → MLP, no message passing) should be run regardless of the gate. The plan lists it as secondary and gated. Our PI-GNN's input embeddings are trainable, which the plan says is exactly the case where C isolates over-parameterisation.

## 8. Gated work (only if A differs from B)

- **Rung C:** `PIGNN` with both SAGE layers replaced by linear layers (same widths, dropout), tuned with the PI-GNN protocol.
- **4A induced preconditioner:** the Jacobian J = ∂Z/∂θ on small graphs (n ≤ a few hundred), K = J Jᵀ at saved checkpoints (weights are saved at every A checkpoint), its spectrum against Â's spectrum and against the spectral profile of good colourings (TabuCol best colourings are saved in `results/tabucol/<g>/seed<s>_best.npy`, DSATUR's can be recomputed).
- **Rung D:** Z ← Z − η M_G ∇_Z L with M_G from 4A (or its linearised form Â X Xᵀ Âᵀ), tuned like B.
- **4B update alignment:** cosine between A's actual logit update ΔZ (from consecutive checkpoints, or a few logged steps), the free-logit gradient −∇_Z L and M_G ∇_Z L, under plain GD and under AdamW.
- **4D architecture checks:** self-loops on/off, orthogonal vs random input features, on a designed bipartite family plus a dense control family.
