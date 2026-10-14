# Experiment 2 — Decoders versus the information in P (RQ3)

**RQ3.** How much of the final result comes from graph-specific information in the learned soft output P, and how much is supplied by the discrete decoder or local search applied afterwards?

In `PI_GNN_Mechanistic_Plan_Final_v3.pdf` this is **Experiment 2** (§5). Started 2026-10-14. Same principles as Experiments 1 and 3: self-contained code, k = χ − 1, graph = statistical unit, 5 seeds, every script through `sbatch`, interpretation rules written before results are seen.

**Status 2026-10-14: done.** PI-GNN outputs (340 runs, all P variants and decoders) and all eight Experiment 3 arms (2,720 runs) decoded; analysis in §6.

**Bottom line (§6).**
1. The decoder adds little: D-CE ≈ argmax + strict descent for PI-GNN (22 better / 46 tie / 0 worse, median −2.8 %) and for every free-logit arm; vertex order and sampling do not matter.
2. Trained P carries real information relative to scrambled, shuffled or untrained P (48–56 graphs better, none worse), but PI-GNN's P is not distinguishable overall from no information at all, i.e. from DSATUR (27 / 28 / 13). This is a family split: PI-GNN better on queen, school, DSJC, flat; DSATUR better on the register-allocation families and DSJR.
3. Decoding does not erase the differences between optimisers: the nine arms keep the same ranking after D-CE (spread down about a tenth). Noisy free logits (σ ≥ 1) beat PI-GNN on most graphs, and σ = 2 is the only arm that beats DSATUR; PI-GNN keeps its small per-graph edge over plain free logits (30 vs 14 graphs after D-CE), although by mean conflicts it ranks 6th of 9 (means are pulled by a few large graphs).

So the final result is set mainly by the optimiser that produced P, not by the decoder, and PI-GNN's P is on average no more useful than what DSATUR obtains from the graph alone.

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

**Arms decoded.** Experiment 1's PI-GNN outputs (`results/main` → `results/exp2/A`, all five variants), and the eight Experiment 3 arms A_nodrop, B, Brand, C, D, B_noise0.5/1/2 (`results/exp3/*` → `results/exp2/<arm>`, variants `trained` and `colperm`; `uniform` is arm-independent and taken from A), for plan conclusion D ("decoding dominates the differences between optimisation methods").

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
| `scripts/analyze_exp2.py` | paired comparisons (Experiment 1's tie rule), family breakdown, arms vs A, spread across arms, cost → `results/analysis/exp2_*.csv` |

## 4. Log

| # | job | what | status |
| --- | --- | --- | --- |
| 1 | 1036669 | unit tests (`test_decoders.py`, `test_controls.py`) | 21/22; the failure was a wrong test (it expected D-CE to copy a one-hot P; it only guarantees no more conflicts) |
| 1b | 1036745 | `test_decoders.py` after fixing that test | 11 passed |
| 2 | 1036675–1036677 | smoke + timing: queen5_5, flat1000_60_0, DSJR500.5 (seed 0) → `results/exp2_test` | passed: 2.7 s, 249 s, 31 s per run |
| 3 | 1036755 | main run, `results/main` epoch 100k, 68 × 5, all variants and decoders → `results/exp2/A/` | 340/340 COMPLETED; trained argmax reproduces Experiment 1's raw and after-descent conflicts on 340/340 runs |
| 4 | 1037265 | `analyze_exp2.py` on arm A only (family breakdown) | done |
| 5 | 1037155–1037162 | the eight Experiment 3 arms, `trained` + `colperm` → `results/exp2/<arm>/` | 8 × 340 COMPLETED |
| 6 | 1039922 | `analyze_exp2.py`, all arms | done; arm-spread comparison wrong (see §6, analysis fix) |
| 7 | 1039926 | `analyze_exp2.py` after the spread fix | done, final numbers in §6 |

## 5. Cost (job 1036755, measured)

- **Wall clock:** submitted 15:25:16, last task ended 15:44:28, **19 min**. Most runs finish in seconds; the tail is the five flat1000 / DSJR500.1c graphs.
- **CPU:** 3.22 CPU-hours over 340 runs; median run 6.6 s, mean 34 s; 42 runs > 60 s, 8 > 300 s, slowest 883 s (flat1000_60_0).
- **By decoder:** sampling + descent of the 32 samples is 2.91 of the 3.22 h (90 %); argmax + descent 0.10 h; D-CE 0.04 h (decode 0.014 h + descent 0.025 h); random-order D-CE 0.03 h.
- **By P variant:** trained 0.24 h; each null 0.68–0.76 h (their colourings start further from a local minimum, so descent takes more moves).
- **Node contention:** flat1000_60_0 took 249 s alone on s-004 (smoke) but 860–880 s on s-005 alongside other tasks; the wall-clock tail is set by this, not by the total CPU.
- **Experiment 3 arms** (jobs 1037155–1037162, `trained` + `colperm`, 8 × 340 = 2,720 runs, all COMPLETED): 15:52:17 → 16:21:56, **30 min** wall clock, **6.9 CPU-h** (0.58–1.17 per arm), throughput limited by ~45 concurrent slots. The `uniform` control does not depend on the arm and is taken from arm A.
- **Extrapolation (written before the arms ran):** an Experiment 3 arm with `trained` + `colperm` costs about (0.24 + 0.69) / 3.22 of this, ≈ 0.9 CPU-h and a similar ~15–20 min wall clock; all eight arms ≈ 7.5 CPU-h. Dropping sampling would cut every figure by about 90 %.

## 6. Results (job 1039926, `scripts/analyze_exp2.py`; descriptive)

Primary metric: conflicts after descent, seed means per graph, 68 graphs, Experiment 1's tie rule (|d| ≤ 5 % or < 1 conflict). The tie rule was adopted from Experiment 1 unchanged after a first quick look at arm A with strict ties; the readings are the same under both. Outputs: `results/analysis/exp2_summary.csv`, `exp2_by_family.csv`, `exp2_arms.csv`, `exp2_arm_spread.csv`, `exp2_cost.csv`.

**PI-GNN's own P (arm A)**

| comparison | better / tie / worse | median d | p | reading |
| --- | --- | --- | --- | --- |
| D-CE vs argmax (trained P) | 22 / 46 / 0 | −0.028 | <1e-4 | not distinguishable (median inside the tie band) |
| trained vs **uniform** (D-CE; = DSATUR) | 27 / 28 / 13 | 0.000 | 0.23 | not distinguishable |
| trained vs colours scrambled (D-CE) | 48 / 20 / 0 | −0.255 | <1e-4 | trained better |
| trained vs rows shuffled (D-CE) | 56 / 12 / 0 | −0.299 | <1e-4 | trained better |
| trained vs untrained GNN (D-CE) | 51 / 14 / 3 | −0.222 | <1e-4 | trained better |
| saturation vs random order (D-CE) | 4 / 64 / 0 | 0.000 | 0.16 | not distinguishable |
| best of 32 samples vs argmax | 2 / 65 / 1 | 0.000 | 0.008 | not distinguishable |
| uniform vs colours scrambled (D-CE) | 41 / 24 / 3 | −0.175 | <1e-4 | uniform better |

Reading against §2: D-CE ≈ argmax + descent, so the decoder gain is mostly simple local improvement. Trained P beats the permuted / untrained P, but those are worse than no information at all (confident and wrong); against the uninformative control (D-CE on uniform = DSATUR) trained P is not distinguishable overall. The plan's table grouped "null or permuted"; the two disagree here, and both are reported.

**By family** (trained vs uniform, D-CE): trained better on queen 12/0/0 (median −0.18), school 2/0/0, DSJC 3/1/0, flat 3/2/0 (−0.05); uniform (DSATUR) better on the register-allocation families fpsol 0/2/1 (+0.57), inithx 0/1/2 (+1.40), miles 0/2/3 (+1.00), zeroin 0/3/0 (+0.20) and on DSJR 0/1/2 (+0.44); le splits 7/0/5; SocialNet, mulsol, myciel tie. The overall tie is a family split, not a uniform null.

**Experiment 3 arms through the same decoders**

| arm | mean after argmax + descent | mean after D-CE + descent | rank (argmax → D-CE) | vs A after D-CE | D-CE vs uniform (DSATUR) |
| --- | --- | --- | --- | --- | --- |
| B_noise2 | 83.4 | 83.4 | 1 → 1 | 35 / 29 / 4 better | 36 / 27 / 5, better |
| B_noise1 | 92.5 | 92.3 | 2 → 2 | 31 / 30 / 7 better | 36 / 25 / 7, p = 0.08 |
| Brand | 107.1 | 106.2 | 3 → 3 | 14 / 25 / 29 | tie |
| B | 108.6 | 107.3 | 4 → 4 | 14 / 24 / 30 | tie |
| B_noise0.5 | 109.2 | 108.4 | 5 → 5 | 13 / 24 / 31 | tie |
| A (PI-GNN) | 114.9 | 111.4 | 6 → 6 | — | tie |
| A_nodrop | 115.7 | 117.0 | 7 → 8 | 4 / 34 / 30 worse | 9 / 36 / 23 |
| C (no message passing) | 120.4 | 116.1 | 8 → 7 | 2 / 44 / 22 | tie |
| D (linearised) | 130.4 | 123.1 | 9 → 9 | 1 / 21 / 46 worse | worse |

- Every arm's trained P beats its own colour-scrambled P (39–58 graphs better, ≤ 1 worse).
- D-CE ≈ argmax + descent for every free-logit arm (63–68 ties); it helps the GNN-type arms a little (A 22, C 27, D 45 graphs better).
- Spread across the nine arms per graph, (max − min) / mean: median 0.87 after argmax, 0.68 after D-CE; smaller under D-CE on 44 graphs, tie 16, larger 8 (median relative change −0.10, p = 2e-7). The ordering of the arms is unchanged.

Reading against §2's last row: arms differ after argmax **and still differ after D-CE** (same ranking, spread down about a tenth), so decoding does not erase the differences between optimisation methods; plan conclusion D is not supported. The optimiser that produces P matters more than the decoder applied to it.

**Analysis fix during this step.** The first full analysis (1039922) compared the per-graph arm spreads with the conflict tie rule, whose "< 1 conflict" clause made every spread (a fraction below 1) a tie (1/67/0). Fixed with a `conflicts=False` mode (5 % relative band only) and rerun as 1039926; no other number changed.
