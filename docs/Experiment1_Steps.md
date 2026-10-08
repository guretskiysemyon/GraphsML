# Experiment 1 — Endpoint dynamics over training length (RQ1)

**Goal.** Test whether PI-GNN's decoded colouring approaches a single-flip local optimum as training proceeds, and how endpoint quality changes with training budget.

**RQ1.** Does PI-GNN approach a hard colouring that is single-flip optimal as training proceeds? How do endpoint quality and local optimality change with training budget?

**Important implementation principle.** This experiment should be self-contained. Do **not** depend on code from other repositories. External repositories such as geocolor or GraphExperiment may be consulted as references or used to compare old results, but every component needed for this experiment should be defined and implemented independently from its algorithmic description.

---

## 0. What is already available, and what still has to be decided

Already available:

- DIMACS graph files.
- A table of chromatic numbers / proven χ values.
- Previous PI-GNN results that can be used only as context, not as a code dependency.

Before the final run:

1. **Resolve the primary graph list.**
   - The previous plan referred to 48 DIMACS graphs with proven χ.
   - The current χ table may contain more eligible graphs.
   - Do not arbitrarily take 48. First determine whether the old 48 were selected by an explicit rule.
   - If there is a clear historical 48-graph benchmark, keep it for direct comparability.
   - Otherwise, use all eligible DIMACS graphs with proven χ and χ ≥ 3, and document that the benchmark was expanded.

2. **Choose one PI-GNN configuration by tuning on a separate tuning set.**
   - The test DIMACS graphs must not be used to choose hyperparameters.
   - Candidate configurations from earlier work may be used as starting points for the tuning grid, but the implementation itself is independent.

3. **Choose checkpoint budgets only after a small pilot.**
   - Example budgets: 100, 1,000, 10,000 epochs, plus the longest feasible run.
   - These are examples, not fixed requirements.

---

## 1. Experimental setting

### 1.1 Primary test set

Use DIMACS graphs with:

- proven chromatic number χ;
- χ ≥ 3.

For every graph record:

- graph name / family;
- number of vertices `n`;
- number of edges `m`;
- χ;
- primary number of colours `k = χ - 1`;
- density;
- maximum degree;
- number of isolated vertices.

### 1.2 Primary regime: `k = χ - 1`

The main experiment uses

`k = χ - 1`.

Why:

- At `k = χ`, a successful zero-conflict colouring is automatically single-flip optimal.
- At `k = χ - 1`, zero conflicts are impossible, so every method must end somewhere in a non-trivial conflict landscape.
- This makes the question "does PI-GNN approach a one-flip local minimum?" meaningful.

Use `k = χ` only as a secondary analysis later.

### 1.3 Tuning / diagnostic set

Generate a separate synthetic set, disjoint from the DIMACS test set, using several graph families such as:

- Erdős–Rényi;
- Watts–Strogatz;
- random regular;
- planted k-colourable graphs.

Try to roughly cover the size and density range of the DIMACS benchmark.

Avoid tuning only on trivial instances where simple heuristics solve nearly everything.

---

## 2. Algorithms and objects that must be implemented independently

Nothing in this section should require importing code from geocolor, GraphExperiment, or another previous repository.

### 2.1 PI-GNN

Implement the per-instance PI-GNN directly from its mathematical definition.

For a graph `G=(V,E)`:

1. Build node features `X`.
2. A GNN `f_theta(G,X)` outputs logits `Z` of shape `n x k`.
3. Convert logits to a soft colouring `P = softmax(Z)` row-wise.
4. Optimise the Potts relaxation:

   `L_Potts(P) = sum_{(u,v) in E} p_u^T p_v`.

5. Save `P` at the selected training checkpoints.

For Experiment 1, the important requirement is not a particular software implementation but that the chosen PI-GNN architecture and optimiser are fixed after tuning and then used consistently on the test set.

### 2.2 Hard argmax decoding

Given `P`, define the hard colouring:

`c_v = argmax_j P[v,j]`.

This is the hard output used for the primary endpoint analysis.

### 2.3 Random colouring

For every vertex, sample a colour uniformly from `{1, ..., k}`.

Use the same number of random seeds as for PI-GNN when comparing the local minima reached after vertex descent.

### 2.4 Fixed-k DSATUR

Implement DSATUR independently:

1. Repeatedly choose an uncoloured vertex with the largest saturation degree (number of distinct colours among already-coloured neighbours).
2. Break ties using ordinary degree, then a deterministic or seeded tie-break rule.
3. Assign the lowest / best available colour among the allowed `k` colours.
4. If all `k` colours create conflicts, assign a colour with the smallest number of conflicts with already-coloured neighbours.

The exact tie-breaking rule must be recorded so results are reproducible.

### 2.5 Randomised DSATUR

Use the same fixed-k DSATUR rule but randomise selected tie-breaks with a seeded RNG.

This is useful as a stronger classical reference, but it is not essential to the central RQ1 claim.

### 2.6 TabuCol — optional quality reference

TabuCol is **not required to run Experiment 1**.

If used, implement it independently from the algorithmic description rather than importing another repository.

At a high level:

- keep exactly `k` colours;
- repeatedly recolour conflicted vertices;
- allow non-improving moves;
- maintain a tabu tenure preventing immediate reversal of recent moves;
- use aspiration when a tabu move gives a new best solution.

Its role here is only to provide a strong reference conflict count at `k = χ - 1`, because the true minimum number of conflicts is generally unknown.

Never call the difference from TabuCol a "gap to optimum".

---

## 3. Shared endpoint-analysis primitives

Implement these first and unit-test them on small hand-checkable graphs such as an odd cycle with `k=2` and `K4` with `k=3`.

### 3.1 Conflict counter

For a hard colouring `c`, count monochromatic edges:

`C(c) = |{(u,v) in E : c_u = c_v}|`.

Each conflicting edge is counted once.

### 3.2 Neighbour-colour count matrix

Maintain:

`M[v,c] = |{u in N(v): colour(u)=c}|`.

Then the number of conflicts incident to vertex `v` under its current colour is `M[v,c_v]`.

If `v` were recoloured to `c'`, its incident conflict count would become `M[v,c']`.

This makes single-flip checking and vertex descent efficient.

### 3.3 Single-flip optimality

A hard colouring is single-flip optimal if **no recolouring of one vertex strictly lowers the total number of conflicts**.

Vertex `v` has an improving move iff:

`min_c M[v,c] < M[v,c_v]`.

Only vertices incident to at least one conflict need to be checked.

Record:

- whether the whole colouring is single-flip optimal;
- number / fraction of conflicted vertices that have an improving move;
- best available one-flip improvement.

The binary flag is the headline measure; the extra quantities tell us *how far* a non-optimal endpoint is from one-flip stability.

### 3.4 Strict-improvement vertex descent

Starting from a hard colouring:

1. Find all strict single-vertex improvements.
2. Choose the move with the largest conflict reduction.
3. Break ties randomly using a seeded RNG.
4. Apply the move and update the neighbour-colour counts.
5. Repeat until no strict improvement exists.

Do **not** use sideways / plateau moves in the primary version.

Record:

- conflicts before descent;
- conflicts after descent;
- number of improving moves;
- fraction of conflicts removed;
- descent wall-clock time.

The final colouring must, by construction, be single-flip optimal.

---

## 4. Optional soft-space diagnostic

This is useful for explaining the PI-GNN trajectory, but it is secondary to the hard single-flip analysis.

For each non-isolated vertex `v` and colour `c`, define:

`q(v,c) = sum_{u in N(v)} p(u,c)`.

The current expected local Potts contribution is:

`sum_c p(v,c) q(v,c)`,

while the best single-coordinate response would choose the colour with minimum `q(v,c)`.

Define:

`g(v) = sum_c p(v,c) q(v,c) - min_c q(v,c) >= 0`.

Normalise by degree:

`g_tilde(v) = g(v) / deg(v)`.

Also compute normalized row entropy:

`H_tilde(v) = - sum_c p(v,c) log p(v,c) / log k`.

Interpret the pair jointly:

- `g_tilde ≈ 0`, low `H_tilde`: near-hard and coordinate-wise stationary / best-response-stable;
- `g_tilde ≈ 0`, `H_tilde ≈ 1`: near the uniform symmetric stationary point;
- larger `g_tilde`: an immediate coordinate-wise improvement is still available in the relaxed problem.

Important: `g_tilde` is a **first-order stationarity diagnostic**, not a certificate of a local or global minimum.

### Sanity checks

For one-hot `P`, `g(v)` equals:

`current incident conflicts - best incident conflicts after one recolouring`.

Therefore `g(v) > 0` exactly when that hard vertex has a strict improving recolouring.

For uniform `P`:

`g_tilde(v)=0` and `H_tilde(v)=1`.

---

## 5. One endpoint-evaluation function

Build a single function conceptually like:

```text
evaluate_endpoint(G, P, k)
```

that returns all endpoint diagnostics for one saved soft output `P`:

1. argmax colouring;
2. conflict count;
3. single-flip-optimal flag;
4. number / fraction of improvable conflicted vertices;
5. best one-flip improvement;
6. strict vertex-descent result;
7. optional per-vertex `(g_tilde, H_tilde)` data;
8. timing.

This function should know nothing about how `P` was produced. That makes it reusable for PI-GNN checkpoints and later experiments.

---

## 6. Unit tests before training anything

Do not wait until the end to test the harness.

### Required tests

- conflict counter on hand-labelled small graphs;
- single-flip checker against exhaustive brute-force one-vertex recolouring;
- incremental neighbour-colour counts against recomputation from scratch;
- vertex descent never increases conflicts;
- vertex descent always terminates at a single-flip-optimal colouring;
- fixed random seeds reproduce tie-breaking;
- one-hot `P`: soft gap agrees with the hard single-flip improvement;
- uniform `P`: `g_tilde = 0`, `H_tilde = 1`.

For small graphs, exhaustive checks are preferable to cross-checking another repository.

---

## 7. Tune PI-GNN and choose training budgets

### 7.1 Tune only on the synthetic tuning set

Tune parameters such as:

- learning rate;
- hidden dimension;
- optimiser;
- dropout;
- any architecture-specific parameters that are part of the chosen PI-GNN implementation.

Record the whole tuning grid and criterion.

Freeze the final configuration **before evaluating the DIMACS test set**.

### 7.2 Choose checkpoint budgets with a pilot

Run the tuned model on a small diagnostic subset and measure:

- time per epoch;
- Potts-loss trajectory;
- conflicts over training;
- whether endpoint metrics are still changing.

Then choose a logarithmic set of checkpoints, for example:

`100, 1,000, 10,000, T_max`.

Do not force 100,000 epochs if it is computationally unrealistic.

### 7.3 Checkpoint semantics

Prefer one long training trajectory and save `P` along it.

The training schedule must be **budget-independent in its prefix**: the state at epoch 1,000 in the long run should represent the same optimization procedure as a run that stopped at epoch 1,000.

A predetermined learning-rate schedule is fine; it does not have to be constant.

---

## 8. Main PI-GNN run

For each primary DIMACS graph:

- use `k = χ - 1`;
- run 5 seeds for final results;
- save `P` at every selected checkpoint;
- record cumulative training wall-clock time;
- record Potts loss.

Exploratory runs may use fewer seeds.

At each checkpoint call the endpoint evaluator from Section 5.

Store:

- graph metadata;
- seed;
- budget;
- Potts loss;
- conflicts before descent;
- single-flip status;
- number / fraction of improvable conflicted vertices;
- best one-flip improvement;
- conflicts after descent;
- number of descent moves;
- fraction of conflicts removed;
- descent time;
- optional `(g_tilde, H_tilde)` data.

A long-format results table is convenient.

---

## 9. Classical reference endpoints

These controls ask whether PI-GNN reaches unusually good local minima or merely ordinary ones.

At the same graph and same `k = χ - 1`:

1. **Random start -> strict vertex descent.**
2. **Fixed-k DSATUR -> strict vertex descent.**
3. **Randomised DSATUR -> strict vertex descent** if useful.
4. **TabuCol conflict reference** only if available / implemented.

The important comparison is not "which method is the best solver?" but:

> Given different starting procedures, what kinds of one-flip local minima are reached?

---

## 10. Optional use of old geocolor / previous outputs

This experiment has **no dependency** on geocolor or another previous repository.

If old saved outputs are easily accessible, they may be used as an optional pilot:

- load the saved `P` matrices;
- pass them through the independently implemented endpoint evaluator;
- use the result only as preliminary evidence.

If only hard colourings were saved, only the hard conflict / single-flip / descent analysis can be performed.

If the outputs are unavailable, skip this step entirely.

Do not delay Experiment 1 to recover old artefacts.

---

## 11. Analysis

### Statistical unit

The graph is the statistical unit.

For a graph with several PI-GNN seeds:

1. compute seed-level metrics;
2. average / summarize them within the graph;
3. compare graphs using paired graph-level results.

Because DIMACS is a heterogeneous benchmark rather than an IID graph sample, treat p-values as descriptive evidence.

### Main analyses

For every training budget report:

- fraction of PI-GNN argmax outputs that are single-flip optimal;
- conflict count before and after strict vertex descent;
- number of improving descent moves;
- fraction of conflicted vertices with an improving move;
- best one-flip improvement;
- endpoint quality relative to random / DSATUR descent;
- optional joint `(g_tilde, H_tilde)` distribution.

Where sample sizes permit, stratify by:

- graph family;
- `n`;
- density;
- baseline difficulty.

TabuCol, if used, is only a quality reference. It is not the definition of success.

---

## 12. Figure 1

The primary figure should make the training-length story immediately visible.

### Main panels

**Panel A — Local optimality versus training budget**

x-axis: training budget  
y-axis: fraction of argmax outputs that are single-flip optimal

**Panel B — Remaining value of simple local search**

For each budget show:

- conflicts before vertex descent;
- conflicts after vertex descent.

Optionally also show:

- number of descent moves;
- fraction of conflicted vertices with an improving move.

**Panel C — Soft-space companion diagnostic**

Show the joint distribution of `(g_tilde, H_tilde)` at several budgets.

This panel is explanatory, not the headline result.

Add random / DSATUR descent reference levels where useful.

---

## 13. Interpretation rules

Write the interpretation before looking at the final test results.

| Observation | Interpretation |
|---|---|
| Single-flip-optimal fraction rises with budget and descent gains vanish | PI-GNN's decoded outputs increasingly approach ordinary one-flip local minima |
| Substantial descent gains remain after long training | PI-GNN often stops before reaching even a basic one-flip endpoint |
| PI-GNN + descent reaches similar endpoints to random/DSATUR + descent | PI-GNN may mainly provide another starting point for simple local search |
| PI-GNN + descent reaches consistently better local minima | PI-GNN is steering the search toward better regions of the discrete landscape |
| `g_tilde ≈ 0` with low entropy | soft output is near-hard and coordinate-wise stationary |
| `g_tilde ≈ 0` with high entropy | soft output is near a symmetric soft stationary point |

### Required caveats

- The Potts local-minimum lemma is mathematical motivation; Experiment 1 is **not** proving it.
- The lemma concerns local minima in `P`-space.
- PI-GNN optimises parameters `theta`, so stationarity in parameter space does not imply a local minimum in `P`-space.
- Argmax can change the endpoint structure.
- `g_tilde` measures coordinate-wise stationarity, not global quality.
- Do not claim that the Potts/single-flip connection itself is new.

---

## 14. Secondary regime: `k = χ`

Only after the primary `k = χ - 1` experiment is complete, optionally repeat the analysis at `k = χ`.

Report:

- success rate (zero-conflict solutions);
- endpoint metrics for failed runs;
- differences between successful and unsuccessful PI-GNN trajectories.

A successful `k = χ` output is automatically single-flip optimal, so it should not dominate the RQ1 endpoint analysis.

---

## 15. Optional TabuCol and warm-start hook

If TabuCol is implemented or saved TabuCol colourings are available:

- check whether those colourings at `k = χ - 1` are already single-flip optimal;
- compare their endpoint quality with PI-GNN;
- use this later to interpret the warm-start no-op case study.

This is optional and should not delay the main experiment.

---

## 16. Recommended implementation order

1. Resolve the graph set and create the benchmark manifest.
2. Implement the DIMACS loader and graph metadata.
3. Implement conflict counting and neighbour-colour counts.
4. Implement and test the single-flip checker.
5. Implement and test strict vertex descent.
6. Implement the optional `g_tilde, H_tilde` diagnostics.
7. Build `evaluate_endpoint(G, P, k)`.
8. Implement random and fixed-k DSATUR controls.
9. Implement PI-GNN independently and verify it on small / tuning graphs.
10. Tune PI-GNN on the disjoint synthetic tuning set.
11. Run a pilot to choose checkpoint budgets.
12. Run the full DIMACS experiment at `k = χ - 1`.
13. Run the random / DSATUR endpoint controls.
14. Analyse and produce Figure 1.
15. Add TabuCol, `k = χ`, old geocolor outputs, or warm-start analysis only if useful.

---

## 17. What is actually required for RQ1

### Required

- DIMACS graphs with known χ;
- independent PI-GNN implementation;
- saved `P` checkpoints;
- argmax decoding;
- conflict counter;
- single-flip checker;
- strict vertex descent;
- random / DSATUR reference endpoints;
- 5 final seeds;
- analysis over training budget.

### Useful but secondary

- `g_tilde, H_tilde`;
- randomised DSATUR;
- TabuCol quality reference;
- `k = χ` secondary run;
- old geocolor / B5 outputs.

### Not needed for Experiment 1

- D-CE;
- free logits;
- preconditioners;
- kernel / spectral analysis;
- q-descent;
- importing code from another repository.
