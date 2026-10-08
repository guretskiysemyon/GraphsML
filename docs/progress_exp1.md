# Experiment 1 — progress log

Last updated: 2026-10-08.

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
- [ ] **F. Budget pilot** on a small subset, using the frozen config: epoch time, loss and conflict trajectories, whether endpoint metrics are still changing; then choose the log-spaced checkpoints.
- [ ] **G. Controls:** random → descent, fixed-k DSATUR → descent (randomised DSATUR if useful).
- [ ] **H. Full run:** DIMACS at k = χ − 1, 5 seeds, all checkpoints.
- [ ] **I. Analysis and Figure 1** (single-flip-optimal fraction vs budget; conflicts before/after descent; the (g̃, H̃) companion panel); interpretation rules written before looking at test results.
- [ ] **J. Only if time allows:** k = χ secondary regime, TabuCol, old geocolor/B5 outputs, warm-start hook.
