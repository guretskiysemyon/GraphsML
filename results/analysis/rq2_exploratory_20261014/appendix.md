## Exploratory appendix: family, structure, repair, and trajectory analyses

*Added 14 October 2026 at the user's request. These analyses were performed after inspecting the main results. They supplement the completed study and do not replace its prespecified metrics or conclusions.*

### E.1 Scope, definitions, and verification

The main exploratory analysis uses all nine arms at exactly 100,000 steps: 68 graphs × five seeds × nine arms = 3,060 endpoints. We analyze both raw conflicts and conflicts after identical strict descent. Unless explicitly identified as run or vertex counts, comparisons use five-seed means within graphs and give each graph equal weight. Register-allocation graphs are pooled for display, but within-family adjustment uses the original 14 manifest families; SocialNet, Mycielski, and the single “other” graph remain separate. The pooling is descriptive, not a claim that all register-allocation graphs are independent examples of one distribution.

Four analyses were completed: (1) structural predictors of paired performance; (2) repair gains, conflict concentration, newly available recolorings, and sensitivity to descent ties; (3) confidence and actual assignment changes over training; and (4) overlap between PI-GNN and noisy-logit benefits. We additionally examine the genuinely extended 13-graph subset separately.

The endpoint aggregates reproduce the existing common-budget per-graph CSV. Every saved final logit tensor was decoded with the original float32-softmax argmax convention. A vectorized implementation of the repository's steepest strict descent reproduces all 3,060 raw counts, repaired counts, descent-move counts, and initial improvable-vertex counts; every reproduced endpoint is single-flip optimal. This decoding detail matters near uniform probabilities, where rounding can change argmax ties. No training was rerun and no original result files were overwritten.

Structural tests comprise 14 features × eight alternatives versus A × two endpoint metrics = 224 associations. Overall correlations are Spearman rank correlations. Within-family correlations subtract original-family means from globally ranked features and outcomes; their two-sided permutation p-values use 4,999 shuffles restricted within original families. Benjamini–Hochberg q-values are computed separately across the 224 overall tests and 224 family-adjusted tests. Diagnostic scans have their own correction; robustness checks and small-family correlations are descriptive. These corrections help organize a post-hoc screen, not convert it into confirmatory evidence. Graph properties are correlated, families are small and heterogeneous, and no new graph set is used for validation.

Define the repaired A advantage as $d_A=(B-A)/\max(A,1)$, with graph-level repaired counts: positive values favor A. Other alternative-versus-A effects use $d_X=(X-A)/\max(A,1)$, so positive values favor A there too. Practical ties retain the paper's rule: absolute difference below one conflict **or** relative difference at most 5%.

### E.2 Family outcomes and structural context

The tables below report alternative wins / practical ties / A wins. They include every alternative, before and after repair, rather than only the families favoring A.

**Raw conflicts at 100k steps.**

| Family | Graphs | A_nodrop | B | Brand | B_noise0.5 | B_noise1 | B_noise2 | C | D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DSJC | 4 | 1/0/3 | 3/0/1 | 3/0/1 | 3/0/1 | 3/0/1 | 3/0/1 | 2/0/2 | 2/0/2 |
| DSJR | 3 | 1/1/1 | 2/0/1 | 2/0/1 | 2/0/1 | 2/1/0 | 3/0/0 | 2/0/1 | 2/0/1 |
| SocialNet | 5 | 0/2/3 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 1/4/0 | 0/5/0 | 0/3/2 |
| flat | 5 | 1/1/3 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 2/3/0 | 2/0/3 |
| le | 12 | 0/1/11 | 6/3/3 | 6/4/2 | 6/3/3 | 9/3/0 | 10/1/1 | 0/5/7 | 0/0/12 |
| miles | 5 | 2/1/2 | 3/2/0 | 3/1/1 | 3/1/1 | 3/2/0 | 4/1/0 | 3/2/0 | 3/1/1 |
| myciel | 5 | 1/2/2 | 0/5/0 | 0/5/0 | 1/4/0 | 1/4/0 | 1/4/0 | 0/5/0 | 0/5/0 |
| other | 1 | 0/0/1 | 0/1/0 | 0/1/0 | 0/1/0 | 0/1/0 | 0/1/0 | 0/1/0 | 0/1/0 |
| queen | 12 | 0/0/12 | 1/4/7 | 1/3/8 | 1/3/8 | 7/4/1 | 10/2/0 | 0/3/9 | 0/0/12 |
| register | 14 | 4/1/9 | 10/3/1 | 6/2/6 | 10/4/0 | 14/0/0 | 14/0/0 | 11/3/0 | 5/2/7 |
| school | 2 | 0/0/2 | 2/0/0 | 2/0/0 | 2/0/0 | 2/0/0 | 2/0/0 | 0/1/1 | 0/0/2 |

**Repaired conflicts at 100k steps.**

| Family | Graphs | A_nodrop | B | Brand | B_noise0.5 | B_noise1 | B_noise2 | C | D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DSJC | 4 | 1/0/3 | 3/0/1 | 3/0/1 | 3/0/1 | 3/0/1 | 3/0/1 | 1/1/2 | 0/1/3 |
| DSJR | 3 | 1/1/1 | 2/0/1 | 2/0/1 | 2/0/1 | 2/1/0 | 3/0/0 | 2/0/1 | 0/0/3 |
| SocialNet | 5 | 0/5/0 | 0/5/0 | 0/4/1 | 0/5/0 | 0/5/0 | 1/4/0 | 0/5/0 | 0/3/2 |
| flat | 5 | 0/4/1 | 4/1/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 0/4/1 | 0/1/4 |
| le | 12 | 0/9/3 | 3/4/5 | 6/3/3 | 5/2/5 | 9/3/0 | 10/1/1 | 0/6/6 | 0/0/12 |
| miles | 5 | 1/3/1 | 2/3/0 | 1/3/1 | 2/1/2 | 2/3/0 | 4/1/0 | 2/2/1 | 0/1/4 |
| myciel | 5 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/3/2 |
| other | 1 | 0/1/0 | 0/1/0 | 0/1/0 | 0/1/0 | 0/1/0 | 0/1/0 | 0/1/0 | 0/1/0 |
| queen | 12 | 0/0/12 | 1/3/8 | 1/3/8 | 1/3/8 | 7/4/1 | 10/2/0 | 0/4/8 | 0/0/12 |
| register | 14 | 3/11/0 | 0/3/11 | 0/1/13 | 0/2/12 | 5/8/1 | 6/7/1 | 0/6/8 | 0/0/14 |
| school | 2 | 1/1/0 | 2/0/0 | 2/0/0 | 2/0/0 | 2/0/0 | 2/0/0 | 0/1/1 | 0/0/2 |

**Structural context: medians across graphs within each display group.**

| family_group | Graphs | median_n | median_k | density | degree_CV | clustering | clique_lb_over_k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DSJC | 4 | 125.000 | 29.500 | 0.699 | 0.057 | 0.700 | 0.708 |
| DSJR | 3 | 500.000 | 84.000 | 0.472 | 0.275 | 0.764 | 0.959 |
| SocialNet | 5 | 87.000 | 10.000 | 0.080 | 1.124 | 0.653 | 1.100 |
| flat | 5 | 300.000 | 27.000 | 0.484 | 0.043 | 0.483 | 0.407 |
| le | 12 | 450.000 | 14.000 | 0.089 | 0.327 | 0.300 | 1.071 |
| miles | 5 | 128.000 | 30.000 | 0.260 | 0.461 | 0.745 | 1.033 |
| myciel | 5 | 47.000 | 5.000 | 0.218 | 0.437 | 0.000 | 0.400 |
| other | 1 | 120.000 | 8.000 | 0.089 | 0.113 | 0.445 | 1.125 |
| queen | 12 | 98.000 | 10.000 | 0.298 | 0.080 | 0.382 | 1.080 |
| register | 14 | 211.000 | 30.000 | 0.164 | 1.071 | 0.668 | 1.033 |
| school | 2 | 368.500 | 13.000 | 0.247 | 0.463 | 0.538 | 1.077 |

These results separate broad failure from conditional advantage: D loses repaired comparisons across most families, while A versus plain B depends on family. SocialNet/Mycielski outcomes often have only one or a few conflicts, making the absolute practical-tie threshold especially relevant. Full family effects, including relative and edge-normalised differences, are in [family_comparisons.csv](/Users/semyon_guretskiy/Documents/Code/GraphsML/results/analysis/rq2_exploratory_20261014/family_comparisons.csv).

### E.3 Which structural properties track A's repaired advantage?

We measure size $n$, palette size $k$, density, mean degree, degree coefficient of variation (standard deviation / mean), maximum-degree / mean-degree ratio, average clustering, transitivity, mean edge-neighborhood Jaccard, mean edge-neighborhood cosine similarity, a deterministic greedy clique lower bound, that lower bound divided by $k$, maximum core number divided by $k$, and mean degree divided by $k$. The clique statistic is a lower bound produced by greedy construction from at most 64 highest-degree starting vertices, **not** the maximum clique. Neighborhood similarities use open neighborhoods. Isolated vertices are retained in clustering averages and all graph counts are checked against the manifest.

The central structural screen uses $d_A$: positive correlation means increasing the feature is associated with greater relative advantage for A after repair.

| feature | rho | q_global | within_family_rho | q_within_family |
| --- | --- | --- | --- | --- |
| n | -0.054 | 0.722 | -0.130 | 0.323 |
| density | -0.500 | 5.98e-05 | -0.648 | 0.008 |
| k | -0.053 | 0.725 | -0.628 | 0.003 |
| mean_degree | -0.458 | 2.95e-04 | -0.742 | 0.001 |
| degree_cv | 0.593 | 9.58e-07 | 0.447 | 0.036 |
| max_degree_over_mean | 0.578 | 1.70e-06 | 0.431 | 0.047 |
| clustering | 0.113 | 0.464 | -0.344 | 0.083 |
| transitivity | -0.155 | 0.286 | -0.337 | 0.047 |
| edge_jaccard | -0.073 | 0.644 | -0.330 | 0.063 |
| neighbour_cosine | -0.014 | 0.923 | -0.296 | 0.112 |
| greedy_clique_lb | 0.048 | 0.751 | -0.629 | 0.002 |
| clique_lb_over_k | 0.139 | 0.348 | 0.280 | 0.133 |
| max_core_over_k | -0.485 | 1.07e-04 | -0.091 | 0.676 |
| mean_degree_over_k | -0.531 | 1.55e-05 | -0.186 | 0.357 |

The strongest overall signals are degree heterogeneity ($\rho=0.593$) and maximum-degree concentration ($\rho=0.578$), favoring A as they increase, and density ($\rho=-0.500$) and mean degree ($\rho=-0.458$), favoring B as they increase. Several relationships survive family adjustment, but the predictors are not independent mechanisms. The proposed overall neighborhood-similarity explanation has little direct support: edge Jaccard has $\rho=-0.073$ and cosine similarity $\rho=-0.014$. Within-family Jaccard is more negative, but its adjusted q-value is about 0.063. Graph size and palette size have little overall association, while palette size has a substantial negative association within families. That contrast warns against interpreting an aggregate near-zero correlation as universal irrelevance.

**Degree-heterogeneity tertiles, using feature quantiles rather than outcome-selected thresholds.**

| feature_min | feature_max | graphs | A wins | ties | B wins | Median A advantage |
| --- | --- | --- | --- | --- | --- | --- |
| 0.009 | 0.146 | 23 | 9 | 5 | 9 | 0.006 |
| 0.152 | 0.556 | 22 | 5 | 9 | 8 | -0.016 |
| 0.575 | 1.757 | 23 | 12 | 11 | 0 | 0.667 |

The most heterogeneous third contains 12 A wins, 11 ties, and no B wins. However, degree heterogeneity is not a validated selection rule: excluding register-allocation graphs reduces its correlation from 0.593 to 0.290; adjustment for ranked size, palette size, and density reduces it to 0.228. Across all leave-one-display-family-out checks, its correlation ranges from 0.290 to 0.706. Excluding graphs where both repaired means are below five leaves 44 graphs and a correlation of 0.441.

**Sensitivity to scale and covariates.** The feature itself is omitted from its control set.

| Feature | Relative effect rho | Edge-normalised effect rho | Excluding both endpoints <5 | Partial rho, size/palette/density |
| --- | --- | --- | --- | --- |
| degree_cv | 0.593 | 0.126 | 0.441 | 0.228 |
| density | -0.500 | -0.280 | -0.563 | -0.619 |
| mean_degree | -0.458 | -0.602 | -0.683 | -0.310 |
| edge_jaccard | -0.073 | -0.206 | -0.315 | -0.039 |
| neighbour_cosine | -0.014 | -0.181 | -0.266 | 0.030 |

Degree heterogeneity has only $\rho=0.126$ with the difference in conflicts per edge, versus 0.593 with relative conflicts. Density also changes strength across summaries. Thus, the observed regime concerns relative endpoint quality, especially where conflict totals are small; it is not evidence of large edge-normalised gains or a unique structural cause. Symmetric relative normalisation gives the same Spearman ordering here, but edge normalisation asks a different question.

**Complete overall structural screen.** Each entry is Spearman rho for alternative versus A; positive values mean that increasing the feature favors A. Adjusted q-values and family-centered versions for every entry are retained in [structure_correlations.csv](/Users/semyon_guretskiy/Documents/Code/GraphsML/results/analysis/rq2_exploratory_20261014/structure_correlations.csv).

**Raw outcome correlations.**

| feature | A_nodrop | B | Brand | B_noise0.5 | B_noise1 | B_noise2 | C | D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n | -0.156 | -0.301 | -0.308 | -0.302 | -0.435 | -0.465 | -0.414 | -0.127 |
| density | -0.117 | -0.302 | -0.339 | -0.288 | -0.209 | -0.202 | -0.223 | -0.231 |
| k | -0.466 | -0.696 | -0.554 | -0.720 | -0.894 | -0.890 | -0.841 | -0.359 |
| mean_degree | -0.247 | -0.577 | -0.591 | -0.582 | -0.625 | -0.611 | -0.609 | -0.283 |
| degree_cv | -0.204 | -0.071 | 0.156 | -0.089 | -0.223 | -0.212 | -0.155 | 0.047 |
| max_degree_over_mean | -0.166 | -0.027 | 0.192 | -0.054 | -0.172 | -0.147 | -0.107 | 0.090 |
| clustering | -0.349 | -0.373 | -0.184 | -0.400 | -0.568 | -0.587 | -0.642 | -0.195 |
| transitivity | -0.249 | -0.397 | -0.340 | -0.384 | -0.436 | -0.502 | -0.523 | -0.177 |
| edge_jaccard | -0.303 | -0.441 | -0.353 | -0.431 | -0.498 | -0.554 | -0.588 | -0.212 |
| neighbour_cosine | -0.337 | -0.462 | -0.349 | -0.459 | -0.547 | -0.592 | -0.621 | -0.239 |
| greedy_clique_lb | -0.419 | -0.591 | -0.450 | -0.615 | -0.820 | -0.837 | -0.797 | -0.292 |
| clique_lb_over_k | 0.304 | 0.491 | 0.391 | 0.503 | 0.489 | 0.446 | 0.251 | 0.283 |
| max_core_over_k | 0.211 | 0.077 | -0.109 | 0.104 | 0.205 | 0.198 | 0.191 | 0.054 |
| mean_degree_over_k | 0.209 | 0.067 | -0.105 | 0.087 | 0.201 | 0.207 | 0.213 | 0.082 |

**Repaired outcome correlations.**

| feature | A_nodrop | B | Brand | B_noise0.5 | B_noise1 | B_noise2 | C | D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n | -0.410 | -0.054 | -0.103 | -0.046 | -0.412 | -0.337 | 0.071 | 0.072 |
| density | -0.125 | -0.500 | -0.500 | -0.506 | -0.305 | -0.317 | -0.368 | -0.409 |
| k | -0.347 | -0.053 | -0.060 | -0.043 | -0.636 | -0.565 | 0.056 | 0.291 |
| mean_degree | -0.541 | -0.458 | -0.489 | -0.475 | -0.646 | -0.540 | -0.193 | -0.205 |
| degree_cv | -0.032 | 0.593 | 0.664 | 0.618 | 0.139 | 0.166 | 0.374 | 0.639 |
| max_degree_over_mean | -0.028 | 0.578 | 0.642 | 0.581 | 0.164 | 0.211 | 0.363 | 0.589 |
| clustering | -0.181 | 0.113 | 0.107 | 0.135 | -0.303 | -0.343 | 0.144 | 0.323 |
| transitivity | 0.060 | -0.155 | -0.172 | -0.111 | -0.356 | -0.436 | -0.069 | 0.129 |
| edge_jaccard | 0.009 | -0.073 | -0.089 | -0.036 | -0.339 | -0.409 | -0.012 | 0.191 |
| neighbour_cosine | -0.015 | -0.014 | -0.026 | 0.021 | -0.348 | -0.401 | 0.045 | 0.242 |
| greedy_clique_lb | -0.289 | 0.048 | 0.026 | 0.063 | -0.599 | -0.560 | 0.108 | 0.399 |
| clique_lb_over_k | 0.192 | 0.139 | 0.071 | 0.166 | 0.266 | 0.203 | 0.072 | -0.012 |
| max_core_over_k | -0.170 | -0.485 | -0.550 | -0.518 | -0.095 | -0.080 | -0.243 | -0.589 |
| mean_degree_over_k | -0.246 | -0.531 | -0.567 | -0.566 | -0.123 | -0.084 | -0.292 | -0.599 |

Feature-by-family correlations are available in [structure_within_groups.csv](/Users/semyon_guretskiy/Documents/Code/GraphsML/results/analysis/rq2_exploratory_20261014/structure_within_groups.csv) for display groups containing at least four graphs; they are descriptive and should not be ranked by small-group p-values. All continuous-feature bins and robustness checks are in [structural_bins.csv](/Users/semyon_guretskiy/Documents/Code/GraphsML/results/analysis/rq2_exploratory_20261014/structural_bins.csv) and [structure_robustness.csv](/Users/semyon_guretskiy/Documents/Code/GraphsML/results/analysis/rq2_exploratory_20261014/structure_robustness.csv).

### E.4 How much repair helps, and how it helps

The following counts describe strict improvement within a run; the between-method practical-tie threshold is not applied. All free-logit variants are unchanged by repair at this budget.

| arm | improved_runs | unchanged_runs | raw_mean | repaired_mean | mean_removed | median_removed_if_helped | graphs_any_help |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 197 | 143 | 1517.347 | 114.865 | 1402.482 | 9.000 | 51 |
| A_nodrop | 283 | 57 | 1558.371 | 115.729 | 1442.641 | 23.000 | 66 |
| B | 0 | 340 | 108.615 | 108.615 | 0.000 | — | 0 |
| Brand | 0 | 340 | 107.071 | 107.071 | 0.000 | — | 0 |
| B_noise0.5 | 0 | 340 | 109.215 | 109.215 | 0.000 | — | 0 |
| B_noise1 | 0 | 340 | 92.474 | 92.474 | 0.000 | — | 0 |
| B_noise2 | 0 | 340 | 83.412 | 83.412 | 0.000 | — | 0 |
| C | 201 | 139 | 133.147 | 120.415 | 12.732 | 4.000 | 64 |
| D | 196 | 144 | 172.562 | 130.447 | 42.115 | 13.000 | 49 |

A and A-no-dropout lose 92.4% and 92.6% of their aggregate raw conflicts through repair, compared with 9.6% for C and 24.4% for D. These percentages are weighted by raw conflict totals and can be dominated by severe failures; they are not typical per-run reductions. Among improved runs, median removals are 9 for A, 23 for A-no-dropout, 4 for C, and 13 for D.

**Fraction of runs helped by repair, by family.**

| family_group | A | A_nodrop | C | D |
| --- | --- | --- | --- | --- |
| DSJC | 55.0% | 95.0% | 75.0% | 85.0% |
| DSJR | 66.7% | 100.0% | 86.7% | 100.0% |
| SocialNet | 16.0% | 56.0% | 12.0% | 20.0% |
| flat | 100.0% | 100.0% | 100.0% | 100.0% |
| le | 61.7% | 88.3% | 91.7% | 100.0% |
| miles | 76.0% | 80.0% | 28.0% | 24.0% |
| myciel | 16.0% | 16.0% | 40.0% | 4.0% |
| other | 0.0% | 40.0% | 20.0% | 0.0% |
| queen | 15.0% | 91.7% | 51.7% | 86.7% |
| register | 97.1% | 94.3% | 44.3% | 7.1% |
| school | 100.0% | 100.0% | 100.0% | 100.0% |

**Median across graphs of seed-mean fraction of raw conflicts removed.**

| family_group | A | A_nodrop | C | D |
| --- | --- | --- | --- | --- |
| DSJC | 30.9% | 48.9% | 8.2% | 22.6% |
| DSJR | 96.5% | 78.4% | 10.2% | 13.1% |
| SocialNet | 0.0% | 26.4% | 6.7% | 0.0% |
| flat | 17.9% | 51.7% | 8.9% | 28.7% |
| le | 2.4% | 44.6% | 6.2% | 14.3% |
| miles | 63.9% | 47.7% | 10.0% | 4.0% |
| myciel | 10.0% | 15.0% | 26.0% | 0.0% |
| other | 0.0% | 27.3% | 10.0% | 0.0% |
| queen | 0.0% | 48.2% | 3.7% | 9.0% |
| register | 55.6% | 72.7% | 10.2% | 0.0% |
| school | 7.2% | 38.1% | 10.6% | 27.4% |

Queen and register-allocation graphs illustrate different routes to A’s advantage. Only 15.0% of queen A runs improve under descent, whereas 97.1% of register-allocation A runs improve. In both families A often beats plain B after repair. A single explanation based on “needing more cleanup” therefore does not cover its advantage.

**Trace of repair on improved runs.** The top 10% statistic is the fraction of conflicting-edge endpoint incidences carried by the most conflicted 10% of vertices. “Initially non-improvable” means a vertex whose improving recoloring became available after earlier moves.

| Arm | Improved runs | Median vertices recoloured | Runs recolouring initially non-improvable vertices | Runs recolouring a vertex more than once | Median raw top-10% conflict share |
| --- | --- | --- | --- | --- | --- |
| A | 197 | 2.2% | 56 | 27 | 45.3% |
| A_nodrop | 283 | 8.0% | 52 | 77 | 30.6% |
| C | 201 | 1.6% | 78 | 18 | 36.1% |
| D | 196 | 4.7% | 130 | 52 | 25.9% |

Most improved A and C runs recolour only a small share of vertices (medians 2.2% and 1.6%). Nonetheless, repair is not always a one-pass cleanup: initially non-improvable vertices are recoloured in 56 A runs, 78 C runs, and 130 D runs. D has a particularly substantial cascading component. These traces describe how descent unfolds; they do not establish that cascades cause better final minima. Conflict concentration alone is also insufficient: already-optimal B outputs can concentrate their few residual conflicts strongly while admitting no improving move. Full run, graph, family, and correlation tables are in [repair_trace_runs.csv](/Users/semyon_guretskiy/Documents/Code/GraphsML/results/analysis/rq2_exploratory_20261014/repair_trace_runs.csv), [repair_trace_graphs.csv](/Users/semyon_guretskiy/Documents/Code/GraphsML/results/analysis/rq2_exploratory_20261014/repair_trace_graphs.csv), [repair_trace_by_family.csv](/Users/semyon_guretskiy/Documents/Code/GraphsML/results/analysis/rq2_exploratory_20261014/repair_trace_by_family.csv), and [repair_trace_correlations.csv](/Users/semyon_guretskiy/Documents/Code/GraphsML/results/analysis/rq2_exploratory_20261014/repair_trace_correlations.csv).

**A distinct raw-output failure mode: one-colour collapse.**

| Arm | One-colour runs | Share of total raw conflicts |
| --- | --- | --- |
| A | 4 | 45.6% |
| A_nodrop | 15 | 83.8% |
| B | 0 | 0.0% |
| Brand | 0 | 0.0% |
| B_noise0.5 | 0 | 0.0% |
| B_noise1 | 0 | 0.0% |
| B_noise2 | 0 | 0.0% |
| C | 0 | 0.0% |
| D | 0 | 0.0% |

A’s four collapsed outputs are all DSJR500.5, seeds 0, 1, 2, and 4; they alone account for 45.6% of A’s total raw conflicts. A-no-dropout has 15 collapsed outputs accounting for 83.8% of its raw conflicts. No such output occurs for C, D, or any free-logit arm. This helps explain why raw aggregate means are so different despite less dramatic graph-wise comparisons. It also shows that neural failures are heterogeneous rather than uniformly saturated.

| arm | graph | seed | k | raw | repaired | frac_pmax_099 | normalised_entropy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_nodrop | queen7_7 | 2 | 6 | 476 | 26 | 1.000 | 6.06e-11 |
| A_nodrop | queen7_7 | 3 | 6 | 476 | 28 | 0.000 | 1.000 |
| A_nodrop | queen8_8 | 0 | 8 | 728 | 17 | 0.000 | 1.000 |
| A_nodrop | DSJC125.1 | 3 | 4 | 736 | 45 | 0.000 | 1.000 |
| A_nodrop | DSJC250.9 | 2 | 71 | 27897 | 74 | 0.000 | 1.000 |
| A_nodrop | flat300_26_0 | 2 | 25 | 21633 | 178 | 0.000 | 1.000 |
| A_nodrop | le450_5b | 4 | 4 | 5734 | 595 | 0.000 | 1.000 |
| A_nodrop | le450_5d | 1 | 4 | 9757 | 552 | 0.000 | 1.000 |
| A_nodrop | le450_5c | 1 | 4 | 9803 | 506 | 0.000 | 1.000 |
| A_nodrop | le450_5c | 4 | 4 | 9803 | 521 | 0.000 | 1.000 |
| A | DSJR500.5 | 0 | 121 | 58862 | 64 | 0.000 | 1.000 |
| A | DSJR500.5 | 1 | 121 | 58862 | 60 | 0.000 | 1.000 |
| A | DSJR500.5 | 2 | 121 | 58862 | 61 | 0.000 | 1.000 |
| A | DSJR500.5 | 4 | 121 | 58862 | 50 | 0.000 | 1.000 |
| A_nodrop | DSJR500.5 | 0 | 121 | 58862 | 64 | 0.000 | 1.000 |
| A_nodrop | DSJR500.5 | 1 | 121 | 58862 | 60 | 0.000 | 1.000 |
| A_nodrop | DSJR500.5 | 2 | 121 | 58862 | 61 | 0.000 | 1.000 |
| A_nodrop | DSJR500.5 | 4 | 121 | 58862 | 50 | 0.000 | 1.000 |
| A_nodrop | DSJR500.1c | 2 | 84 | 121275 | 51 | 0.000 | 1.000 |

Eighteen of these 19 collapsed outputs have entropy at least 99% of the uniform entropy and no vertices with $p_{\max}\geq0.99$; one A-no-dropout queen7_7 output is confidently collapsed. Thus, almost-uniform probabilities and saturated incorrect assignments are distinct endpoint phenomena.

### E.5 Is the repaired ranking sensitive to random tie-breaking?

For each saved raw coloring we evaluate ten new descent tie seeds, 10000–10009, and average their conflict counts rather than selecting the best. Outputs already single-flip optimal are reused unchanged. This yields 30,600 tie-seed outcomes, with 8,770 new descents on the 877 improvable outputs. The original run-seed descent remains the reference.

| arm | variable_tie_runs | mean_tie_range | mean_original_conflicts | mean_tie_conflicts |
| --- | --- | --- | --- | --- |
| A | 104 | 5.165 | 114.865 | 114.470 |
| A_nodrop | 199 | 9.879 | 115.729 | 116.017 |
| B | 0 | 0.000 | 108.615 | 108.615 |
| B_noise0.5 | 0 | 0.000 | 109.215 | 109.215 |
| B_noise1 | 0 | 0.000 | 92.474 | 92.474 |
| B_noise2 | 0 | 0.000 | 83.412 | 83.412 |
| Brand | 0 | 0.000 | 107.071 | 107.071 |
| C | 75 | 2.597 | 120.415 | 120.313 |
| D | 130 | 9.071 | 130.447 | 130.561 |

**Paired comparisons after averaging tie-breaks.** Counts favor X versus Y.

| X | Y | Outcome | X_better | ties | Y_better | median_d |
| --- | --- | --- | --- | --- | --- | --- |
| A_nodrop | A | Original descent | 7 | 40 | 21 | 0.087 |
| B | A | Original descent | 17 | 25 | 26 | 0.060 |
| B_noise2 | A | Original descent | 44 | 21 | 3 | -0.275 |
| C | A | Original descent | 5 | 35 | 28 | 0.078 |
| D | A | Original descent | 0 | 10 | 58 | 0.366 |
| A_nodrop | A | Mean of ten descents | 8 | 38 | 22 | 0.086 |
| B | A | Mean of ten descents | 16 | 26 | 26 | 0.060 |
| B_noise2 | A | Mean of ten descents | 42 | 23 | 3 | -0.280 |
| C | A | Mean of ten descents | 4 | 36 | 28 | 0.065 |
| D | A | Mean of ten descents | 0 | 8 | 60 | 0.390 |

Individual endpoints can depend on tie-breaking: conflict counts vary in 104 A runs, 199 A-no-dropout runs, 75 C runs, and 130 D runs. Nevertheless, the central ordering is stable. Plain B versus A changes from 17/25/26 to 16/26/26; C versus A changes from 5/35/28 to 4/36/28; D still has no wins; noise-2 remains strongly favorable at 42/23/3. This makes a lucky single descent tie seed an inadequate explanation of the main results. Ten samples do not exhaust all possible descent paths. Per-graph changed verdicts and family-specific sensitivity are retained in [repair_sensitivity_verdicts.csv](/Users/semyon_guretskiy/Documents/Code/GraphsML/results/analysis/rq2_exploratory_20261014/repair_sensitivity_verdicts.csv) and [repair_sensitivity_changed_graphs.csv](/Users/semyon_guretskiy/Documents/Code/GraphsML/results/analysis/rq2_exploratory_20261014/repair_sensitivity_changed_graphs.csv).

### E.6 Confidence, actual assignment changes, and endpoint quality

High confidence is not equivalent to a fixed coloring. We measure both confidence from saved probability diagnostics and actual changes between saved colorings. For label-invariant comparison, we optimally align color labels between checkpoints using a maximum-weight assignment; the reported partition-change fraction counts vertices still different after that alignment. Edge-status changes count edges switching between conflicting and nonconflicting. These are endpoint-to-endpoint changes and cannot recover intervening switches.

**First saved checkpoint with at least 90% of vertices having $p_{\max}\geq0.99$.** Step medians are across runs that reach the threshold; never-reaching runs are counted separately. Checkpoint spacing makes these observed crossing times, not exact event times.

| arm | runs | ever_confident | median_first_step | stable_confident | median_stable_step | reversal_runs |
| --- | --- | --- | --- | --- | --- | --- |
| A | 340 | 282 | 200.000 | 274 | 200.000 | 22 |
| A_nodrop | 340 | 275 | 100.000 | 252 | 100.000 | 45 |
| B | 340 | 295 | 5000.000 | 295 | 5000.000 | 1 |
| B_noise0.5 | 340 | 295 | 3000.000 | 295 | 3000.000 | 3 |
| B_noise1 | 340 | 290 | 2000.000 | 290 | 2000.000 | 5 |
| B_noise2 | 340 | 188 | 1000.000 | 182 | 1000.000 | 17 |
| Brand | 340 | 295 | 2000.000 | 295 | 2000.000 | 0 |
| C | 340 | 309 | 100.000 | 289 | 100.000 | 23 |
| D | 340 | 291 | 1000.000 | 288 | 1000.000 | 9 |

**Changes from 50k to 100k steps: median across graph-level seed means.**

| arm | vertex_switch_fraction | partition_switch_fraction | edge_conflict_status_switch |
| --- | --- | --- | --- |
| A | 40.9% | 29.8% | 0.8% |
| A_nodrop | 64.5% | 48.4% | 1.6% |
| B | 0.0% | 0.0% | 0.0% |
| B_noise0.5 | 0.0% | 0.0% | 0.0% |
| B_noise1 | 21.0% | 19.3% | 0.3% |
| B_noise2 | 27.7% | 27.2% | 0.3% |
| Brand | 0.0% | 0.0% | 0.0% |
| C | 25.1% | 25.1% | 0.7% |
| D | 8.0% | 8.0% | 0.3% |

Although A and C become confident early, their late color partitions still change substantially: the median aligned changed-vertex fractions are 29.8% for A and 25.1% for C. Plain B and B-rand have median zero late changes, while noise-2 has 27.2%. Therefore, “early confidence means the neural coloring freezes” is not supported. High confidence can coexist with continued exploration or switches; late movement by itself does not guarantee better endpoints. Relatively few edges change conflict status compared with the number of changed vertices, so assignment movement and conflict-count progress should be distinguished.

**Late aligned partition changes by family.**

| family_group | A | A_nodrop | B | B_noise2 | C | D |
| --- | --- | --- | --- | --- | --- | --- |
| DSJC | 18.2% | 27.0% | 0.0% | 14.9% | 11.4% | 6.6% |
| DSJR | 31.5% | 13.7% | 0.0% | 26.2% | 17.7% | 5.7% |
| SocialNet | 57.8% | 59.2% | 0.0% | 51.7% | 61.7% | 16.8% |
| flat | 14.7% | 40.1% | 0.0% | 10.5% | 16.5% | 4.8% |
| le | 11.2% | 54.9% | 0.0% | 23.8% | 18.8% | 5.5% |
| miles | 57.8% | 50.9% | 0.0% | 35.2% | 52.5% | 7.3% |
| myciel | 38.3% | 33.3% | 0.0% | 48.1% | 40.6% | 4.3% |
| other | 43.0% | 38.2% | 0.0% | 36.0% | 32.3% | 4.8% |
| queen | 8.6% | 58.3% | 0.0% | 5.6% | 12.0% | 8.4% |
| register | 35.2% | 40.3% | 0.0% | 53.8% | 52.7% | 22.5% |
| school | 20.6% | 43.7% | 2.6% | 11.4% | 25.1% | 8.0% |

**Confidence at step 100 versus final repaired conflicts per edge.** Positive rho means more early confidence is associated with worse final edge-normalised outcomes.

| arm | rho | within_family_rho | partial_rho_n_k_density |
| --- | --- | --- | --- |
| A | 0.715 | 0.711 | 0.247 |
| A_nodrop | 0.840 | 0.718 | 0.543 |
| B | 0.629 | 0.709 | -0.160 |
| Brand | 0.848 | 0.897 | 0.585 |
| B_noise0.5 | 0.613 | 0.686 | -0.113 |
| B_noise1 | 0.574 | 0.661 | -0.339 |
| B_noise2 | 0.498 | 0.638 | -0.378 |
| C | 0.806 | 0.670 | 0.457 |
| D | 0.516 | 0.771 | -0.262 |

Every arm shows a positive overall and family-centered association, but controlling ranked size, palette size, and density substantially weakens or reverses it for several arms: A falls from 0.715 to 0.247, B changes from 0.629 to −0.160, and noise-2 from 0.498 to −0.378. Early confidence is consequently a descriptive regime marker, not evidence that commitment causes failure. The structural adjustment is itself an exploratory linear adjustment on ranks, not a causal model.

**Confidence at vertices that admit a strict improving recoloring at 100k.** These are pooled run-vertex counts, not independent statistical samples.

| Arm | Improvable run-vertices | Also confident | Confident share | Runs with confident improvable vertices |
| --- | --- | --- | --- | --- |
| A | 18851 | 13713 | 72.7% | 193 |
| A_nodrop | 38080 | 32847 | 86.3% | 269 |
| C | 2369 | 2363 | 99.7% | 201 |
| D | 5863 | 5788 | 98.7% | 195 |

For C, 99.7% of initially improvable run-vertices are already above the 0.99 confidence threshold. This directly demonstrates saturated but locally improvable outputs without message passing. A has both confident residual errors and almost-uniform collapses, producing a lower pooled confident share. The diagnostic is a mismatch between softmax confidence and discrete improvement, not a proof of convergence failure in parameter space. All diagnostic associations, including family adjustment and normalised palette usage, are in [diagnostics_family_adjusted.csv](/Users/semyon_guretskiy/Documents/Code/GraphsML/results/analysis/rq2_exploratory_20261014/diagnostics_family_adjusted.csv); trajectory and assignment summaries by family are in [trajectory_by_family.csv](/Users/semyon_guretskiy/Documents/Code/GraphsML/results/analysis/rq2_exploratory_20261014/trajectory_by_family.csv) and [assignment_switch_by_family.csv](/Users/semyon_guretskiy/Documents/Code/GraphsML/results/analysis/rq2_exploratory_20261014/assignment_switch_by_family.csv).

### E.7 Do PI-GNN and logit noise help on the same graphs?

Define benefits relative to the same plain-B baseline:

$$b_A=\frac{B-A}{\max(B,1)},\qquad b_{\mathrm{noise}}=\frac{B-B_{\mathrm{noise}}}{\max(B,1)}.$$

Positive values mean fewer repaired conflicts than B. This avoids comparing two effects both divided by A, but the benefits still share a baseline and are not independent observations. We also examine edge-normalised benefits.

| noise | normalisation | n | rho | p | within_family_rho | within_family_p |
| --- | --- | --- | --- | --- | --- | --- |
| B_noise1 | B | 68 | 0.334 | 0.005 | -0.139 | 0.375 |
| B_noise1 | edges | 68 | 0.337 | 0.005 | 0.424 | 0.003 |
| B_noise2 | B | 68 | 0.110 | 0.370 | -0.177 | 0.239 |
| B_noise2 | edges | 68 | 0.198 | 0.105 | 0.357 | 0.031 |

Noise-1 has moderate overall overlap with A’s relative benefit ($\rho=0.334$), but it vanishes after within-family adjustment ($\rho=-0.139$). Noise-2’s overall relative-benefit correlation is only 0.110; the within-family value is −0.177. Edge normalisation yields different within-family associations (0.424 for noise-1 and 0.356 for noise-2). The overlap therefore depends on scale and grouping and does not identify a common mechanism.

**Noise-2 versus B, stratified by the repaired B-versus-A verdict.** “B better” means plain B beats A; “A better” means A beats B.

| B versus A | Noise-2 beats B | Noise-2 ties B |
| --- | --- | --- |
| B better | 16 | 1 |
| A better | 25 | 1 |
| Practical tie | 12 | 13 |

Noise-2 beats B on 25 of the 26 graphs where A beats B and on 16 of the 17 graphs where B beats A, with no practical losses to B. It therefore helps beyond the subset where PI-GNN has an advantage. Family-specific noise-benefit correlations are in [noise_overlap_correlations.csv](/Users/semyon_guretskiy/Documents/Code/GraphsML/results/analysis/rq2_exploratory_20261014/noise_overlap_correlations.csv); with only 2–14 graphs per group, perfect or large small-group correlations should not be treated as mechanistic evidence.

### E.8 Longer training on the same selected graphs

The earlier mixed final-budget result preserves the overall ranking, but this does not mean individual methods stop improving. Here we isolate the 13 graphs with A extended from 100k to 300k and compare each graph with itself at the two budgets. Only graph-arm pairs with all five seeds at both checkpoints enter the table. Counts below compare 300k against 100k within the same method; they are not A-versus-B wins.

| arm | metric | graphs | better_at_300k | ties | worse_at_300k | mean100k | mean300k | median_d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | raw | 13 | 10 | 2 | 1 | 3702.046 | 514.092 | -0.117 |
| A | repaired | 13 | 6 | 6 | 1 | 145.692 | 133.877 | -0.127 |
| B | raw | 13 | 0 | 13 | 0 | 126.138 | 126.138 | 0.000 |
| B | repaired | 13 | 0 | 13 | 0 | 126.138 | 126.138 | 0.000 |
| Brand | raw | 13 | 0 | 13 | 0 | 124.646 | 124.646 | 0.000 |
| Brand | repaired | 13 | 0 | 13 | 0 | 124.646 | 124.646 | 0.000 |
| B_noise0.5 | raw | 13 | 0 | 13 | 0 | 128.277 | 128.277 | 0.000 |
| B_noise0.5 | repaired | 13 | 0 | 13 | 0 | 128.277 | 128.277 | 0.000 |
| B_noise1 | raw | 13 | 7 | 6 | 0 | 97.215 | 88.662 | -0.125 |
| B_noise1 | repaired | 13 | 7 | 6 | 0 | 97.215 | 88.662 | -0.125 |
| B_noise2 | raw | 13 | 5 | 8 | 0 | 85.508 | 81.615 | -0.058 |
| B_noise2 | repaired | 13 | 5 | 8 | 0 | 85.508 | 81.615 | -0.058 |
| D | raw | 2 | 2 | 0 | 0 | 182.600 | 160.000 | -0.147 |
| D | repaired | 2 | 1 | 1 | 0 | 130.400 | 122.800 | -0.069 |

A improves its repaired endpoint under the practical rule on six of the 13 selected graphs, ties on six, and worsens on one. Plain B, B-rand, and noise-0.5 are unchanged on all 13; noise-1 improves on seven, and noise-2 on five, with no practical worsening for those noisy arms. Thus, the unchanged aggregate ordering coexists with continuing progress for A and stronger-noise logits. C and A-no-dropout have no 300k coverage. D has complete paired coverage for only two extended graphs and must not be compared as if it covered the same 13. Selection of the extension set limits generalisation. Exact graph-level values and coverage are in [extended_subset_graphs.csv](/Users/semyon_guretskiy/Documents/Code/GraphsML/results/analysis/rq2_exploratory_20261014/extended_subset_graphs.csv).

### E.9 What these analyses add to RQ2

The clearest additions are that A’s repaired advantage is associated with structural regime and effect scale, repair tie-breaking does not explain away the main ranking, early confidence does not imply frozen assignments, and confident local errors occur in the MLP as well as PI-GNN. Noise-2 improves plain B across both sides of the A-versus-B split. The data also expose two distinct neural endpoint failures: nearly uniform one-colour collapse and confidently improvable colorings.

The structural evidence motivates a narrower hypothesis—relative neural benefit may be associated with degree heterogeneity and density—but does not establish those properties as causes or a reliable method-selection policy. The weak overall neighborhood-similarity findings do not support elevating that proposed explanation. A small, independently designed graph set varying degree heterogeneity and density while controlling size and palette would be needed to test the structural hypothesis. Current post-hoc findings remain candidates for later inclusion, not new primary conclusions.

![Exploratory structure, benefit overlap, and confidence trajectories](files/rq2_exploratory_overview.png)

*Exploratory overview. Left: degree heterogeneity versus repaired A advantage. Middle: A and noise-2 benefits with a common B baseline. Right: median graph-level seed-mean confidence trajectories. These plots illustrate association and trajectory differences, not causal effects.*

### E.10 Reproducible artifacts and audit

The full outputs are retained in [the exploratory analysis directory](/Users/semyon_guretskiy/Documents/Code/GraphsML/results/analysis/rq2_exploratory_20261014). The three computation scripts and report generator are saved alongside the repository scripts:

- [analyze_rq2_exploratory.py](/Users/semyon_guretskiy/Documents/Code/GraphsML/scripts/analyze_rq2_exploratory.py)
- [analyze_rq2_repair_sensitivity.py](/Users/semyon_guretskiy/Documents/Code/GraphsML/scripts/analyze_rq2_repair_sensitivity.py)
- [analyze_rq2_robustness.py](/Users/semyon_guretskiy/Documents/Code/GraphsML/scripts/analyze_rq2_robustness.py)
- [report_rq2_exploratory.py](/Users/semyon_guretskiy/Documents/Code/GraphsML/scripts/report_rq2_exploratory.py)

Inputs are the graph manifest and `.col` files, the existing Experiment 3 tidy/per-graph/movement CSVs, and the saved checkpoint `Z` tensors. Run the computation scripts in the order above, then the report generator. Required packages are NumPy, pandas, SciPy, NetworkX, PyTorch, and matplotlib; the scripts are CPU-only. No training, best-of selection, or change to the original primary-rule CSVs is performed. SHA-256 hashes of aggregate inputs and script files are saved in [input_hashes.json](/Users/semyon_guretskiy/Documents/Code/GraphsML/results/analysis/rq2_exploratory_20261014/input_hashes.json).

Completed checks: 68 graphs match manifest sizes/edge counts; nine arms each have 340 unique common-budget endpoints; all original endpoint quantities reproduce; all 30,600 tie outcomes are non-worsening and single-flip optimal; all extended-subset means use five seeds at both budgets. Missing/constant correlations are shown as dashes rather than zero. Selected exploratory q-values have a permutation resolution floor of 0.0002; no stronger numerical precision or population-wide interpretation is implied.
