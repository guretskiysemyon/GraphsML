# Experiment 1 — progress log

Last updated: 2026-10-09.

Experiment 1 = endpoint dynamics over training length (RQ1) from `PI_GNN_Mechanistic_Plan_Final_v3.pdf`: after PI-GNN trains at `k = χ − 1`, is its argmax colouring already single-flip optimal, and how much does strict vertex descent still improve it, as a function of training budget?

Full step list: [Experiment1_Steps.md](Experiment1_Steps.md). This file records what has been done, what was decided, and what is next.

---

## 1. Done so far

- [x] Read the whole plan PDF (13 pages) and extracted the Experiment 1 requirements (§4 shared protocol, §5 Experiment 1, §9 Stages 0–1, §10 Figure 1, §13 risks).
- [x] Wrote the step-by-step implementation plan, [Experiment1_Steps.md](Experiment1_Steps.md). It was then revised (by Semyon) into a **self-contained** version: no code is imported from geocolor or GraphExperiment, and TabuCol, old B5 outputs and k = χ are optional or secondary.
- [x] Inspected the data. [data/dimacs/](../data/dimacs/) has 80 `.col` files, and [chi_bounds.csv](../data/dimacs/chi_bounds.csv) has 68 graphs with `chi_proven = True`, all with χ ≥ 3.
- [x] Located the reference PI-GNN implementation and read its method (Section 2).
- [x] Created the Python environment (Section 3).
- [x] Created the code layout: `src/graphsml/` (`dimacs.py`, `model.py`, `losses.py`, `train.py`), `tests/`, `scripts/`, `configs/`, `results/`.
- [x] Implemented the plain-PyTorch PI-GNN (Step A): `dimacs.py` parses `.col` files into a 0-indexed `networkx.Graph`; `model.py` has a from-scratch mean-aggregator `SAGEConvMean` (matches `dgl.nn.SAGEConv(aggregator_type="mean")`'s `W_self·h_self + W_neigh·h_neigh + b`, dense adjacency) and `PIGNN` (embedding → SAGE → ReLU → dropout → SAGE → logits); `losses.py` has the Potts loss `0.5·sum(A ⊙ PPᵀ)`, argmax decoding, and a basic conflict counter; `train.py` is the fixed-budget training loop (no early stopping, eval-mode `P` logged every epoch, the paper's patience/tolerance rule logged but not applied).
- [x] 8 unit tests (`tests/test_losses.py`, `tests/test_model.py`): Potts loss equals conflict count for one-hot `P` (triangle and odd-cycle cases), mean-adjacency row sums (1 for connected nodes, 0 for isolated), SAGE forward shape, isolated-node logits stay finite. All pass.
- [x] `scripts/validate_step_a.py` reproduces Table III (PI-SAGE, `k = χ`) on anna/myciel5/queen5_5 (see decision 9).
- [x] **Stage 2 tuning (long-budget confirmation, Slurm array 995824) finished 2026-10-08.** All 780 units (10 configs × 13 tuning graphs × 2 regimes × 3 seeds, 50,000 epochs each) completed cleanly. Summarised with `scripts/summarize_tuning.py --dir results/tuning/stage2` → `results/tuning/stage2_summary.csv`. The stage-1 winner (`stage1_id 28`) is also the stage-2 winner, with a clearer margin over the runner-up at the longer budget (mean rank 4.79 at 10k epochs → 3.17 at 50k epochs vs. 3.63 for the runner-up). This resolves the caveat in decision 7. **Config frozen** and saved to [configs/pignn_frozen.json](../configs/pignn_frozen.json): `lr=0.01962, hidden_dim=30, embed_dim=85, dropout=0.4656`.
- [x] **Step H first sweep finished (Slurm array 1000276, 2026-10-08 17:05–18:57).** 68/68 tasks COMPLETED; all 340 runs (68 graphs × 5 seeds, k = χ − 1, 100,000 epochs) have a DONE marker in `results/main/` (1.4 GB), no errors, all exit codes 0.
- [x] **Step I, first pass: `scripts/analyze_exp1.py` (CPU) via `scripts/analyze_exp1.sbatch`, job 1003734 on s-004, 2026-10-09 10:12:59–10:14:07.** Outputs in `results/analysis/` (`figure1.png`, `sfo_by_family.png`, `figure1_summary.csv`, `per_graph_epoch.csv`, `tidy_seed_level.csv`, `extension_candidates.csv`). What it shows (5-seed means, 100k epochs): single-flip-optimal fraction 0 at epoch 1 → 0.42 at 100k and still rising (17/68 graphs optimal on all seeds, 30/68 on none); strongly family-dependent (queen/myciel/SocialNet ≈ 0.85–1.0 with no descent gain; flat/school/fpsol/inithx/zeroin/mulsol ≈ 0–0.08 with large descent gains); median relative descent gain falls 0.95 → 0.10; the single-flip-optimal fraction at 100k is below its earlier peak on 15 graphs (non-monotone); no graph reaches 0 conflicts after descent (expected at k = χ − 1); 9 graphs still have mean row entropy > 0.5 nats at 100k. **Warning sign motivating step G:** mean conflicts *after descent* barely depends on training (132 at epoch 1 → 115 at 100k; descent from the epoch-1 argmax is already about as good), so the rising single-flip-optimal fraction may only mean the argmax becomes self-consistent, not that it becomes a better start. Known gaps of this first pass: panel C uses unnormalised mean row entropy (nats), not H̃, and no g̃ (neither is in the sweep logs); `paper_stop_epoch` is not exported to a CSV.
- [x] **Step G (controls) done 2026-10-09:** random → descent, fixed-k DSATUR → descent, randomised DSATUR → descent, 68 graphs × 3 methods × 5 seeds (array 1003745, 1020 rows). Protocol and interpretation rules are in §9, results in §10, per-script run log in §8. Headline: PI-GNN (100k epochs) ends with fewer conflicts after descent than a random start on 62/68 graphs (1 loss), but is not distinguishable from deterministic DSATUR overall (16 wins / 28 ties / 24 losses), with a strong family split. The "warning sign" in the first-pass analysis above (mean after-descent conflicts 132 → 115) is **not supported per graph**: trained PI-GNN beats its own epoch-1 argmax on 59/68 graphs (1 loss), median −32%; the means are dominated by the large graphs.

## 2. PI-GNN reference: what we know

**Source.** Official implementation: [amazon-science/gcp-with-gnns-example](https://github.com/amazon-science/gcp-with-gnns-example) (MIT-0 code; DGL, Python 3.7, a notebook plus `minimal_utils.py`). Paper: Schuetz et al., [arXiv:2202.01606](https://arxiv.org/abs/2202.01606).

**Method as published:**

- **Input:** a trainable `nn.Embedding(n, d0)`.
- **Layers:** two GraphSAGE (mean aggregator) or GraphConv layers, in → hidden → k, with ReLU and dropout between them. The paper's configs use K = 1 hidden layer.
- **Loss:** the Potts loss `L = Σ_{(u,v)∈E} p_u · p_v` (the notebook writes it as `sum(A * (P Pᵀ)) / 2`).
- **Decoding:** `argmax(P)` per vertex.
- **Optimiser:** AdamW (PI-SAGE, COLOR graphs) or Adam (PI-GCN, citation graphs), default betas, learning rate per graph.
- **Budget:** up to 100,000 epochs with early stopping (patience 500).
- **Early-stopping rule in the code:** a patience counter increases when `|loss − prev_loss| ≤ tolerance` *or* when the loss increases. With dropout noise this can fire early.
- **Not verified:** the notebook delegates to `minimal_utils.run_gnn_training`; I could not confirm whether it evaluates with dropout off or tracks the best coloring seen.

**Paper settings usable for validation** (Table III, PI-SAGE, k = χ, 100,000 epochs, patience 500):

| graph | q | embedding d0 | hidden | learning rate | dropout |
| --- | --- | --- | --- | --- | --- |
| anna | 11 | 43 | [22] | 0.03507 | 0.3298 |
| myciel5 | 6 | 16 | [18] | 0.01333 | 0.3964 |
| queen5_5 | 5 | 77 | [32] | 0.02988 | 0.3784 |

(The paper tuned hyperparameters per graph. The file names in the DIMACS folder are `anna`, `myciel5`, `queen5_5`.)

**Notebook example config (for reference only):** GraphConv, embedding 64, hidden 64, dropout 0.1, lr 1e-4, 50,000 epochs, tolerance 1e-3. This differs from Table III.

## 3. Environment

- **Machine:** c-006, 4× NVIDIA TITAN Xp (12 GB each), Python 3.12.3.
- **Virtualenv:** `/home/morg/NLP_2526b/semyong/.venvs/graphsml`. Activate with `source /home/morg/NLP_2526b/semyong/.venvs/graphsml/bin/activate`, or call its `bin/python` directly.
- **Packages:** torch 2.7.1+cu126, numpy, scipy, pandas, networkx, matplotlib, pyarrow, pytest, tqdm. Pinned in [requirements.txt](../requirements.txt).
- **Verified:** `torch.cuda.is_available()` is True, with a real matmul on the GPU (compute capability 6.1).
- **Setup note:** `python3 -m venv` could not bootstrap pip (the system has no `ensurepip`), so pip was installed with PyPA's official `get-pip.py`.
- **Not installed:** DGL and PyTorch Geometric (deliberately; see decision 1).

## 4. Decisions made

1. **Implement PI-GNN ourselves in plain PyTorch.** Use the official repo as a reference only. Why: we must change the training loop anyway (fixed-budget checkpoints, eval-mode P, saved initial state, dense logging); later experiments need clean access to parameters and gradients (free logits, preconditioned descent, update alignment, kernel analysis); DGL is pinned to an old Python; and it matches the self-contained principle. The risk is subtle deviations from the published model, handled by the validation in Step A below.
2. **Early stopping off** for Experiment 1. Run to the fixed budget, and only log the epoch where the paper's rule *would* have stopped.
3. **P is saved from an eval-mode forward pass** (dropout off), so it is deterministic. The training-mode loss is logged separately.
4. **Save everything needed now and later, at every checkpoint, for every (graph, seed):**
   - logits Z (P = softmax Z can be derived), plus the **initial logits Z₀**. Experiment 3's free-logit arm must start from exactly Z₀;
   - the **initial model weights** and the model weights at each checkpoint (for parameter-distance, update-alignment and kernel analyses);
   - a trajectory log: training-mode loss, eval-mode loss, argmax conflicts, mean row entropy, number of colours used, wall-clock time, with dense logging early and sparser later;
   - the config, seeds, device and library versions.

   Estimated size is under about 1 GB in total.
5. **Primary regime** `k = χ − 1`; `k = χ` is secondary. Statistical unit is the graph; 5 seeds for final numbers.
6. **Order of work:** validate, then tune on synthetic graphs, then pilot the budgets, and only then run the full DIMACS sweep. Do not train on all DIMACS graphs first.
7. **One frozen PI-GNN configuration** for all graphs (no per-graph tuning), chosen on the synthetic tuning set only (`data/synthetic/tuning/`, 13 graphs; `scripts/make_tuning_set.py`). Stage 1 = random search, 43 configs (3 Table III references + 40 draws) x 13 graphs x 2 regimes (k_hi, k_hi-1) x 2 seeds x 10,000 epochs; score = mean rank of final conflicts (`scripts/summarize_tuning.py`). Submitted as Slurm array 994828 on 2026-10-07. Stage 2 = long-budget confirmation of the top 6 stage-1 configs plus 4 extra draws, 3 seeds x 50,000 epochs, Slurm array 995824, finished 2026-10-08. Stage 1's winner held up as stage 2's winner (see §1) — **frozen, no longer a caveat.**
8. **Checkpoints:** a base log grid (1, 2, 5 x 10^j, ~3 per decade) shared by all graphs for comparability, plus extra checkpoints that grow with n (~3 per decade at n~100 up to ~6 at n~1000). `T_max` per graph from the pilot, not a formula in n. Training must be resumable (save optimiser + RNG state; partition is killable). Evaluate only at checkpoints, not every epoch. Record the device; do not mix CPU and GPU within a comparison; no silent CPU fallback.
9. **Validation result (Step A):** model matches DGL `SAGEConv` to float precision (`scripts/check_dgl_equivalence.py`). Embedding init changed to N(0,1) as in the paper's code. Full-budget runs reproduce the paper on myciel5/6, jean, queen5_5; the harder queen graphs trail the paper's best, consistent with the paper reporting best-of-many (an independent 100-run reproduction, arXiv:2406.00792 App. B, finds e.g. queen7_7 zero conflicts in 1/100 runs). Treated as passed with that caveat.
10. **Free logits (Experiment 3) get their own tuning later**, same protocol and effort on the same synthetic set, starting from the saved `Z₀`; the PI-GNN config is not reused for them. Nothing to do for it in Experiment 1 beyond saving `Z₀` and the initial weights.
11. **Checkpoint trajectory is reported raw, never as a running best-so-far.** PI-GNN's argmax-decoded conflicts/loss are not guaranteed to improve monotonically with more training: dropout makes the training-mode loss itself noisy (the paper's own patience rule checks for loss *increases*, not just plateaus), and argmax decoding is a discontinuous projection of the smooth Potts loss, so a vertex can flip to a worse colour while the aggregate soft loss keeps falling (§13's "argmax can change the endpoint structure" caveat). We already have a concrete example: `validate_pignn_valA_992715_10.out`, queen5_5 seed 2, best eval conflicts over training = 0 but final = 13. Panels A/B must use the raw value *at* each checkpoint, not a monotonic running minimum, or the non-monotonicity — which is itself a possible RQ1 finding — would be masked by construction. `paper_stop_epoch` (logged, not applied) gives a ready-made check for whether stopping early would in fact have beaten running to `T_max`.
12. **"Best of the 5 seeds" is a secondary capacity view, never the headline number**, to avoid reintroducing the paper's own best-of-many bias (decision 9's caveat). Primary RQ1 numbers are always the mean over the 5 seeds, graph as the statistical unit (decision 5). When a "best observed run" is shown, it is selected by lowest *raw* argmax-conflict count at a fixed, checkpoint-matched budget `T` across the 5 seeds (not by soft loss — loss and hard conflicts can diverge, see decision 11 — and not by scanning each seed's own full trajectory for its best-ever epoch, which would silently compare different budgets against each other). The per-seed running-best-so-far value (already tracked as `best_conflicts_so_far`) is kept as a separate, clearly labelled diagnostic ("how good could this run have been with an oracle stopping rule"), not used in the main panels.
13. **First full sweep: all 68 graphs, 5 seeds, `T_max` = 100,000 for everyone** (2026-10-08). The pilot (5 DIMACS graphs, seed 0; note these are test graphs, used for budgets only, not hyperparameters) showed no single budget fits all: queen5_5/jean are flat by ~1k epochs, queen11_11 is single-flip optimal only from 50k, and le450_15a (n=450) and flat1000_50_0 (n=1000) are still improving at 300k and never settle into a single-flip-optimal argmax (flat1000: 3207 conflicts at 100k → 2681 at 300k; descent from an *untrained* epoch-1 argmax already reaches ~1630 vs ~1500 from the 300k endpoint). Size is a poor proxy for convergence, so rather than a per-graph T_max by n, run everything to 100k first and extend unconverged graphs later. Runs live in `results/main/<graph>_seed<s>/` (no budget in the name), so a finished run extends in place to a larger `--t-max` by resuming (tested: extension and a mid-run `kill -9` both reproduce an uninterrupted run exactly). Run time is launch-bound (~4-5 ms/epoch regardless of n, and 5 seeds on one GPU do not slow each other), so one Slurm array task = one graph with its 5 seeds as 5 parallel processes on one GPU (`scripts/run_sweep.sbatch`, order and metadata in `configs/sweep_manifest.csv`, smallest n first). Because Figure 1 compares budgets, once graphs are extended to different budgets the panels must be drawn on a matched graph set per budget, not on whichever graphs reached it.

## 5. Open items

1. **Primary graph list.** (68 graphs with proven chi are listed in `docs/graphs_proven_chi.csv`; use all 68 unless the old 48-graph rule turns up.) The old plan says 48 DIMACS graphs with proven χ; `chi_bounds.csv` has 68. Find out whether the 48 came from an explicit rule. If yes, keep them for comparability; otherwise use all 68 and document that the benchmark was expanded.
2. **Two suggestions on [Experiment1_Steps.md](Experiment1_Steps.md) not applied yet:**
   - §7.3: "a predetermined learning-rate schedule is fine" is only valid if the schedule depends on the epoch alone, not on `T_max` (a cosine decay set to `T_max` would break checkpoint equivalence).
   - §13, row 4: "PI-GNN + descent reaches consistently better local minima → PI-GNN is steering the search toward better regions" claims more than Experiment 1 can show; random/DSATUR descent doesn't isolate the GNN's contribution (that is Experiment 3). Suggested wording: "PI-GNN's argmax is a better start for descent than random/DSATUR".
3. **TabuCol is optional now**, which departs from the plan (it measures quality against the best TabuCol result). Without it, the only quality yardsticks are random-start and DSATUR descent. State this in the report.
4. **Layout** for code and results. Proposal: `src/`, `tests/`, `configs/`, `results/` under `GraphsML/`.
5. **Optional equivalence check** of our SAGE layer against DGL's `SAGEConv` (same weights, compare outputs) in a throwaway environment.

## 6. Risk to watch

At `k = χ − 1`, PI-GNN may stay near the uniform output (H̃ ≈ 1) for a long time. Then argmax is nearly arbitrary, and a low single-flip-optimal fraction would reflect an untrained output, not a finding about local minima. So log the number of colours used and mean row entropy at every checkpoint, to separate "not trained yet" from "trained but stuck".

## 7. Next steps (in order)

- [x] **A. Implement and validate PI-GNN (done; see decision 9).** Plain-PyTorch model (embedding → SAGE → ReLU → dropout → SAGE → softmax), Potts loss over edges, AdamW. Reproduce the paper's zero-conflict results at k = χ with the Table III settings on anna, myciel5 and queen5_5. Optionally compare against DGL (item 5 above).
- [x] **B. Training and logging script — done.** [src/graphsml/checkpoints.py](../src/graphsml/checkpoints.py): `log_checkpoint_grid(t_max, n)` (decision 8's 1-2-5-per-decade base grid, scaled up to 6/decade for large `n`, always a superset of the base grid so budgets stay comparable across graphs), 6 tests in [tests/test_checkpoints.py](../tests/test_checkpoints.py). [src/graphsml/train.py](../src/graphsml/train.py): `train_pignn_checkpointed` — fixed-budget, no early stopping, saves `Z₀`/initial weights once, per-checkpoint `Z`/weights + a raw (never running-min) trajectory row (decision 11), and a resumable `resume_state.pt` (model, optimiser, full RNG state, bookkeeping) so a killed job restarts from its last completed checkpoint instead of epoch 1. 4 tests in [tests/test_train_checkpointed.py](../tests/test_train_checkpointed.py), including a simulated mid-epoch crash that resumes and reproduces an uninterrupted reference run bit-for-bit. Constant learning rate (no `T_max`-dependent schedule), so open item 2's §7.3 concern doesn't apply yet; revisit if a schedule is ever added.
- [x] **C. Harness primitives with tests — done.** [src/graphsml/endpoint.py](../src/graphsml/endpoint.py): `EndpointGraph` (adjacency-list + degree, precomputed once per graph and reused across seeds/checkpoints), `neighbour_colour_counts`/`conflict_count`/`conflicts_from_counts`, `apply_flip` (incremental M update), `single_flip_report`, `strict_vertex_descent` (steepest-improvement, seeded tie-breaks, no sideways moves), `soft_stationarity` (g̃, H̃), and `evaluate_endpoint(eg, P, k, seed)` tying them together. 9 tests in [tests/test_endpoint.py](../tests/test_endpoint.py), all passing: conflict counter on odd-cycle(k=2)/K4(k=3); single-flip report vs. brute-force one-vertex recolouring on 10 random graphs; incremental M vs. recompute-from-scratch over 20 flips; descent never increases conflicts and always ends single-flip-optimal (10 random graphs); same-seed descent is reproducible on a tied (K4) instance; one-hot P's soft gap equals the hard one-flip gain; uniform P gives g̃=0, H̃=1; an `evaluate_endpoint` smoke test.
- [~] **D. Benchmark manifest (graph list in `docs/graphs_proven_chi.csv`, 68 graphs; per-graph metadata table still to be finalised):** resolve the graph list (open item 1) and write per-graph metadata (n, m, χ, k = χ − 1, density, max degree, isolated vertices, family).
- [x] **E. Synthetic tuning set and PI-GNN tuning — done.** Tuning set built (Erdős–Rényi, Watts–Strogatz, random regular, planted k-colourable, 13 graphs). Stage 1 random search (994828) and stage 2 long-budget confirmation (995824) both finished; config frozen in [configs/pignn_frozen.json](../configs/pignn_frozen.json).
- [x] **F. Budget pilot — done 2026-10-08.** `scripts/budget_pilot.py`, results in `results/pilot/` (queen5_5, jean, queen11_11, le450_15a, flat1000_50_0; 100k epochs, plus 300k re-runs of le450_15a and flat1000_50_0). Outcome: ~5 ms/epoch regardless of n; budgets needed span 1k to >300k; decision 13 sets the first sweep at 100k for all graphs.
- [x] **G. Controls — done 2026-10-09:** random → descent, fixed-k DSATUR → descent, randomised DSATUR → descent. See §8 (run log), §9 (protocol), §10 (results). Code: [src/graphsml/controls.py](../src/graphsml/controls.py), [tests/test_controls.py](../tests/test_controls.py) (9 tests, passing), `scripts/run_controls.py`, `scripts/analyze_controls.py` (+ their `.sbatch` files). Open caveat: the randomised-DSATUR variant is probably weaker than intended (§10).
- [~] **H. Full run:** DIMACS at k = χ − 1, 5 seeds, all checkpoints. First pass (100k epochs, all 68 graphs) **done** (array 1000276, see §1). Still to do: extend unconverged graphs in place (decision 13); the first-pass extension rule flags 13 graphs (DSJC125.9, DSJC250.9, DSJR500.1c, flat1000_60_0, flat300_26_0, flat300_28_0, fpsol2.i.3, inithx.i.1, le450_25c, miles1500, school1_nsh, zeroin.i.1, zeroin.i.2); flat1000_50_0 and the 9 high-entropy graphs (see §1) should be considered too. Not submitted yet.
- [~] **I. Analysis and Figure 1:** first pass done (job 1003734, see §1). Still to do: normalised H̃ and g̃ for panel C (needs a pass over the saved checkpoint logits), `paper_stop_epoch` export, the controls overlay, matched graph sets once graphs are extended. Interpretation rules for the controls are in §9, written before the controls were run.
- [ ] **J. Only if time allows:** k = χ secondary regime, TabuCol, old geocolor/B5 outputs, warm-start hook.

## 8. Run log (every script is run through `sbatch`, never directly)

Rule: before a script is submitted its row is added here as "about to submit"; once Slurm reports the result the row is updated. Cluster account `gpu-students`, partition `studentkillable`; CPU-only jobs request no GPU so any node can take them.

| # | script (sbatch file) | what | job id | node / time | status |
| --- | --- | --- | --- | --- | --- |
| 1 | `scripts/run_sweep.py` (`run_sweep.sbatch`) | Step H, 68 graphs × 5 seeds × 100k epochs | array 1000276 | 2026-10-08 17:05–18:57 | done, 68/68 COMPLETED |
| 2 | `scripts/analyze_exp1.py` (`analyze_exp1.sbatch`) | Step I first pass, Figure 1 | 1003734 | s-004, 2026-10-09 10:12:59–10:14:07 | done |
| 3 | `tests/test_controls.py`, `tests/test_endpoint.py` (`run_tests.sbatch`) | unit tests for the new DSATUR / random-start controls (+ the existing endpoint tests they build on) | 1003741 | s-004, 2026-10-09 10:46:30–10:48:12 | done, 18 passed (9 new in `test_controls.py`) |
| 4 | `scripts/run_controls.py` (`run_controls.sbatch`) | Step G: 68 graphs × 3 methods × 5 seeds → descent; array 0-67, 1 CPU each, no GPU | array 1003745 | s-003/s-004/s-005/s-006, 2026-10-09 10:48:21–10:49:31 | done, 68/68 COMPLETED, 1020 rows in `results/controls/*/controls.jsonl` (= 68 × 3 × 5), no errors |
| 5 | `scripts/analyze_controls.py` (`analyze_controls.sbatch`) | PI-GNN vs controls after descent, per the §9 rules; writes `results/analysis/controls_*.csv` and `controls_vs_pignn.png` | 1003814 | s-004, 2026-10-09 10:50:09–10:50:42 | done, exit 0. Numbers are final (recorded in §10); the figure had overlapping labels, so it is regenerated by #6 |
| 6 | `scripts/analyze_controls.py` (`analyze_controls.sbatch`) | rerun after edits to the script: diagonal moved into the legend, labels for the largest absolute differences, counts on narrow bars, and two descriptive columns added (`wins_abs_le5`, `losses_abs_le5`) | 1003815 | s-004, 2026-10-09 10:52:27–10:52:40 | done, exit 0. Per-graph table byte-identical to #5 (diffed against the saved copy). Figure still had overlapping labels (largest-absolute-difference points sit on the diagonal), so superseded by #7 |
| 7 | `scripts/analyze_controls.py` (`analyze_controls.sbatch`) | rerun, figure only: labels now chosen by \|P − C\| × \|log(P/C)\| (far from the diagonal in ratio and magnitude), top 3, counts beside narrow last bars. No change to any number | 1003816 | s-004, 2026-10-09 10:53:22–10:53:35 | done, exit 0. Per-graph table and summary byte-identical to #6. Final figure: `results/analysis/controls_vs_pignn.png` |
| 8 | `scripts/run_sweep.py` (`run_sweep.sbatch`, now takes `T_MAX` from the environment, default 100000) | Step H extension in place, 100k → 300k epochs, the 13 graphs flagged by `extension_candidates.csv` (manifest rows 18, 23, 35, 36, 38, 40, 41, 42, 44, 55, 61, 65, 67), 5 seeds each; `--export=ALL,T_MAX=300000`. The 100k `endpoint_trajectory.jsonl`/`trajectory.jsonl`/`DONE` of these 65 runs were copied to `results/main_100k_snapshot/` first, to check that the rows ≤ 100k are reproduced | — | — | about to submit |

## 9. Step G: protocol and interpretation rules (written 2026-10-09, before any control result was produced)

**What is run.** At k = χ − 1 (same k as the sweep), for every graph and seeds 0–4: (a) a uniformly random k-colouring (RNG `default_rng([seed, 1])`), (b) deterministic fixed-k DSATUR, (c) randomised DSATUR with `default_rng([seed, 2])`. Each start goes through the same `strict_vertex_descent` as PI-GNN's argmax, descent RNG `default_rng(seed)`. DSATUR rule (recorded for reproducibility): vertex = max saturation, then max degree, then lowest index (random among the remaining ties in the randomised version); colour = fewest already-coloured neighbours of that colour, lowest index (random among the minimisers in the randomised version). The deterministic DSATUR gives one start per graph, so its 5 seeds differ only in the descent tie-breaks. Raw starts, single-flip reports and after-descent conflicts go to `results/controls/<graph>/controls.jsonl`.

**Primary comparison.** Per graph, `P_g` = mean over 5 seeds of PI-GNN conflicts after descent at epoch 100,000 (from `results/main`); `C_g` = the same mean for a control. Relative difference `d_g = (P_g − C_g) / max(C_g, 1)`. Per graph: **tie** if `|P_g − C_g| < 1` conflict or `|d_g| ≤ 0.05`; otherwise **win** for PI-GNN if `P_g < C_g`, **loss** if `P_g > C_g`. The graph is the statistical unit, 68 graphs. The primary controls are *random* and *DSATUR (deterministic)*; for each, a two-sided Wilcoxon signed-rank test on the `d_g` (α = 0.025 each, Bonferroni for the two tests), plus median `d_g` and the win/tie/loss counts.

**Secondary / exploratory (no tests, no claims).** Randomised DSATUR; PI-GNN at epoch 1 (an essentially untrained start) as an extra control; per-family breakdown; the raw conflicts of each start before descent.

**Interpretation, fixed in advance (for each primary control):**

| Outcome | Reading |
| --- | --- |
| median `d_g` < −0.05, p < 0.025 and wins > losses | PI-GNN's trained argmax is a better start for strict descent than this control. Wording limit: this does *not* show PI-GNN "steers the search"; isolating the GNN's role is Experiment 3. |
| p ≥ 0.025, or \|median `d_g`\| ≤ 0.05 | PI-GNN's trained argmax is not distinguishable from this control as a start for strict descent (§13 row 3: PI-GNN may mainly be another starting point for simple local search). |
| median `d_g` > +0.05, p < 0.025 and losses > wins | PI-GNN's argmax is a worse start for descent than this control. |

Caveats to state in the report: this compares the local minima reached, not compute (100k training epochs against milliseconds for the controls), and not solver quality (the minima are far above 0 conflicts at k = χ − 1 for most graphs); one frozen PI-GNN config, tuned on synthetic graphs, is compared against untuned controls; per-family patterns are descriptive only.

## 10. Step G results (analysis job 1003814, final figure from 1003816; files in `results/analysis/controls_*`)

All 68 graphs, 5 seeds each, k = χ − 1, PI-GNN at 100,000 epochs against each control, conflicts after strict vertex descent (seed means per graph; verdict and test exactly as fixed in §9).

| control | role | PI-GNN wins | ties | losses | median d | Wilcoxon p | mean conflicts after descent (control / PI-GNN) | reading under the §9 rules |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random start | primary | 62 | 5 | 1 | −0.387 | 5.1e-12 | 134.5 / 114.9 | PI-GNN is a better start for strict descent than a random start |
| DSATUR (deterministic) | primary | 16 | 28 | 24 | 0.000 | 0.0026 | 121.9 / 114.9 | not distinguishable overall (\|median d\| ≤ 0.05) |
| DSATUR (randomised) | exploratory | 29 | 24 | 15 | −0.052 | 0.62 | 120.7 / 114.9 | no claim |
| PI-GNN at epoch 1 | exploratory | 59 | 8 | 1 | −0.325 | 2.1e-12 | 132.1 / 114.9 | no claim |

**What the rules give.** Against random starts PI-GNN is clearly better, which is the low bar. Against deterministic DSATUR the pre-registered reading is "not distinguishable" (§13 row 3), because the median relative difference is 0. The Wilcoxon p (0.0026) is nominally below α = 0.025: the signed differences are large and of both signs, which is the next point.

**Post hoc, descriptive (not pre-registered; do not present as a test).** The DSATUR comparison is a family split, not a tie:
- PI-GNN wins on **queen** (8 of 12 graphs, median d −0.16), **le450_5a–d** (all 4; DSATUR is poor there, e.g. 746 vs 512 on le450_5d), **DSJC125.1/.5**, **flat300_26_0** and **school1_nsh**: graphs where many conflicts remain at k = χ − 1. 12 of its 16 wins are by more than 5 conflicts.
- DSATUR wins on the **register-allocation `.i.` graphs** (fpsol, inithx, zeroin; median d +2.8, +3.6, +1.0; the 5 mulsol graphs are all ties), **miles** (4 of 5), **DSJR** (3 of 3, median d +2.2), homer, and le450_15a/b and le450_25a/c/d. 15 of its 24 wins are by ≤ 5 conflicts (e.g. 2.0 vs 1.0), but 9 are large: DSJR500.5 (59 vs 12), DSJR500.1c (41 vs 13), le450_25c/d (109/105 vs 63), le450_15a/b (72/73 vs 50/48), school1 (466 vs 271), DSJC250.9 (63 vs 51).
- Several of the DSATUR wins (DSJR500.5, DSJR500.1c, DSJC250.9, miles1500, fpsol2.i.1, inithx.i.1, zeroin.i.2) are on the nine graphs with mean row entropy > 0.5 at 100k, and DSJR500.1c, DSJC250.9, miles1500, inithx.i.1 and zeroin.i.2 are also on the extension-candidate list. On the dense ones the raw PI-GNN argmax has orders of magnitude more conflicts than DSATUR's raw colouring (DSJR500.5: 47,161 vs 16; DSJR500.1c: 39,210 vs 28; DSJC250.9: 4,968 vs 65; miles1500: 527 vs 1), so these may be undertrained rather than a property of PI-GNN. The comparison should be repeated for them after extension (step H).
- PI-GNN does beat its own epoch-1 start on 59/68 graphs (median −32%), so training is not irrelevant to what descent reaches.

**Caveats specific to the controls as implemented.**
1. **Deterministic DSATUR depends on the vertex order in the `.col` file** (lowest-index tie-breaks). That may help on structured instances such as the `.i.` graphs; the effect is untested. The randomised variant (below) does worse than deterministic DSATUR on exactly those families (e.g. fpsol2.i.3: 5.0 vs 1.0).
2. **The randomised DSATUR as run is probably weaker than the "stronger classical reference" the plan wants.** In §9 I fixed its colour rule as a uniform choice among *all* minimum-conflict colours; the usual randomised DSATUR randomises only vertex ties and keeps the lowest feasible colour, which wastes fewer colours. Its result (29/24/15, p = 0.62) should therefore not be read as "DSATUR with random ties is no better than PI-GNN". If a stronger randomised reference is wanted, add a variant with random vertex ties and lowest-index colours as a clearly labelled post hoc control rather than altering this one.
3. The comparison is about the local minima reached, not compute or solver quality; at k = χ − 1 no colouring reaches 0 conflicts. Single frozen PI-GNN config vs untuned controls. One seed-mean per graph; a difference of 1 conflict or more counts as a win or loss (under 1 is a tie), so many verdicts on the small-count graphs rest on 1–2 conflicts.
