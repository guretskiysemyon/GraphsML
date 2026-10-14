# Experiment 2 — Decoders versus the information in P (RQ3)

**RQ3.** How much of the final result comes from graph-specific information in the learned soft output P, and how much is supplied by the discrete decoder or local search applied afterwards?

In `PI_GNN_Mechanistic_Plan_Final_v3.pdf` this is **Experiment 2** (§5). Started 2026-10-14. Same principles as Experiments 1 and 3: self-contained code, k = χ − 1, graph = statistical unit, 5 seeds, every script through `sbatch`, interpretation rules written before results are seen.

---

## 1. Design

Nothing is retrained. Each run's P is taken from its saved checkpoint at epoch 100,000 (P = softmax(Z)), and the same set of decoders is applied to it and to null versions of it.

**P variants** (per run = graph × seed):

| variant | what | plan |
| --- | --- | --- |
| `trained` | the run's own P | — |
| `uniform` | softmax of N(0, 0.1²) logits: uniform plus tie-breaking noise | null (a) |
| `untrained` | softmax of the run's `z0.pt` (eval-mode output of the untrained GNN) | null (b) |
| `rowperm` | trained rows permuted across vertices | null (c) |
| `colperm` | each trained row's colour coordinates permuted independently (keeps confidence and entropy, destroys cross-vertex colour coordination) | null (d) |

**Decoders**, each reported raw and after the Experiment 1 strict vertex descent (same code, descent RNG = `default_rng(seed)`):

| decoder | what |
| --- | --- |
| `argmax` | as in Experiment 1 |
| `dce` | D-CE in saturation order (most distinct fixed-neighbour colours, then degree, then index); each vertex fixed to the colour with the fewest expected conflicts given the fixed vertices and the soft rows of the rest. Guarantee: conflicts ≤ L(P). |
| `dce_rand` | D-CE in a random vertex order | null (e) |
| `sample` | K = 32 independent colourings c_v ~ p_v; best and mean, raw and after descent |

Note: D-CE on exactly uniform P is fixed-k DSATUR (unit test), so `uniform` + `dce` is effectively the DSATUR baseline with random tie-breaks.

**Arms decoded.** Experiment 1's PI-GNN outputs (`results/main`, all five variants). Optionally the Experiment 3 arms (`results/exp3/*`, variants `trained` and `colperm`), for plan conclusion D ("decoding dominates the differences between optimisation methods").

## 2. Interpretation rules (fixed before results)

Primary metric: conflicts after descent, per graph averaged over seeds, paired across graphs (win/tie/loss, median difference, Wilcoxon; descriptive, as in Experiments 1 and 3). Raw conflicts are reported alongside.

| result | reading |
| --- | --- |
| D-CE ≈ argmax + descent (trained P) | most of the decoder gain is simple discrete local improvement |
| D-CE(trained) better than argmax + descent **and** than D-CE on `colperm` / `rowperm` / `uniform` | the learned P carries graph-specific information that argmax loses |
| D-CE(trained) ≈ D-CE(null or permuted) | the decoder behaves like the solver; the learned P contributes little |
| D-CE only slightly better but much slower | the sophisticated decoder is not justified computationally |
| Exp 3 arms differ after argmax but not after D-CE | decoding dominates differences between optimisation methods |

Known degenerate case: runs whose trained P collapsed to one colour on every vertex (e.g. DSJR500.5) make `rowperm` identical to `trained`; they are flagged, not dropped.

## 3. Files

| file | what |
| --- | --- |
| `src/graphsml/decoders.py` | D-CE (both orders), sampling, Potts loss, null-P builders |
| `tests/test_decoders.py` | D-CE ≤ L(P), D-CE on one-hot P, D-CE(uniform) = DSATUR, bipartite exactness, sampling frequencies, permutation invariants |
| `scripts/run_decoders.py`, `run_decoders.sbatch` | one array task per run, writes `results/exp2/<arm>/<graph>_seed<s>/decoders.jsonl` + `DONE` with the run's wall time |

## 4. Log

| # | job | what | status |
| --- | --- | --- | --- |
| 1 | 1036669 | unit tests (`test_decoders.py`, `test_controls.py`) | 21/22; the failure was a wrong test (it expected D-CE to copy a one-hot P; it only guarantees no more conflicts) |
| 1b | 1036745 | `test_decoders.py` after fixing that test | 11 passed |
| 2 | 1036675–1036677 | smoke + timing: queen5_5, flat1000_60_0, DSJR500.5 (seed 0) → `results/exp2_test` | passed: 2.7 s, 249 s, 31 s per run |
| 3 | 1036755 | main run, `results/main` epoch 100k, 68 × 5, all variants and decoders → `results/exp2/A/` | 340/340 COMPLETED; trained argmax reproduces Experiment 1's raw and after-descent conflicts on 340/340 runs |

## 5. Cost (job 1036755, measured)

- **Wall clock:** submitted 15:25:16, last task ended 15:44:28, **19 min**. Most runs finish in seconds; the tail is the five flat1000 / DSJR500.1c graphs.
- **CPU:** 3.22 CPU-hours over 340 runs; median run 6.6 s, mean 34 s; 42 runs > 60 s, 8 > 300 s, slowest 883 s (flat1000_60_0).
- **By decoder:** sampling + descent of the 32 samples is 2.91 of the 3.22 h (90 %); argmax + descent 0.10 h; D-CE 0.04 h (decode 0.014 h + descent 0.025 h); random-order D-CE 0.03 h.
- **By P variant:** trained 0.24 h; each null 0.68–0.76 h (their colourings start further from a local minimum, so descent takes more moves).
- **Node contention:** flat1000_60_0 took 249 s alone on s-004 (smoke) but 860–880 s on s-005 alongside other tasks; the wall-clock tail is set by this, not by the total CPU.
- **Extrapolation:** an Experiment 3 arm with `trained` + `colperm` costs about (0.24 + 0.69) / 3.22 of this, ≈ 0.9 CPU-h and a similar ~15–20 min wall clock; all eight arms ≈ 7.5 CPU-h. Dropping sampling would cut every figure by about 90 %.
