# Learned probability information and discrete decoding in physics inspired graph neural network coloring

*RQ3 of Mechanistic Dissection of PI-GNN for Graph Coloring*  
*Results available as of 14 October 2026*

## Abstract

Physics-inspired graph neural networks (PI-GNNs) produce soft color assignments that must be converted into a hard coloring. We investigate whether their final quality comes from information in the learned probabilities or from the discrete decoder and local search applied afterwards. We freeze the 100,000-step outputs on 68 DIMACS graphs, using five seeds and the deliberately infeasible palette size $k=\chi-1$, and compare argmax, derandomized conditional-expectation decoding (D-CE), random-order D-CE, and sampling, before and after identical strict vertex descent. For PI-GNN, D-CE substantially improves raw argmax but adds only a modest overall gain after descent: 22 graph-level wins, 46 practical ties, no losses, and a median relative difference of −2.8%. Learned probabilities outperform independently color-scrambled, vertex-shuffled, and untrained controls after D-CE and descent, with median reductions of 22–30%. However, they have no established overall advantage over a near-uniform, graph-only D-CE control: 27 wins, 28 ties, 13 losses, and zero median difference. This aggregate result conceals strong family differences. Applying the same decoders to eight additional optimization arms reduces, but does not erase, their quality differences; noisy free logits retain their advantage over PI-GNN. Random vertex order and best-of-32 sampling provide little additional benefit for trained PI-GNN outputs, and sampling accounts for about 90% of the recorded PI-GNN decoder-study task time. The results support useful learned coordination relative to corrupted probabilities, without establishing a broad advantage over graph-only decoding or a large contribution from soft information beyond argmax plus elementary repair.

## 1 Introduction

The broader project separates three components of PI-GNN coloring: the optimization procedure producing a soft assignment, the information in that assignment, and the discrete procedure turning it into vertex colors. The companion [RQ1 paper](RQ1_Paper.md) shows that training produces useful starts for strict vertex descent, while many confident outputs still admit improving recolorings. The [RQ2 paper](RQ2_Paper.md) compares the neural parameterization with direct vertex logits and other optimization controls. Here we hold the saved outputs fixed and address the remaining core question:

> **RQ3. How much of the final result comes from graph-specific information in the learned soft output $P$, and how much is supplied by the discrete decoder or local search applied afterwards?**

A graph-aware decoder can improve a coloring even when its probability input contains no learned information. Consequently, a large improvement over raw argmax does not establish that the decoder extracted useful information from the neural output. The relevant comparisons must also give simple repair to argmax and apply the same graph-aware decoder to informative, uninformative, and corrupted probability matrices.

Two distinctions are central. First, learned probabilities can be better than misleading probabilities without improving on a decoder supplied with no trained probabilities. Second, D-CE can act as a discrete improvement procedure even on hard assignments; a gain from D-CE is not automatically evidence that residual probability mass carries useful information beyond argmax. We therefore interpret the decoder and null-input comparisons jointly.

The subquestions below are ordered by their importance to answering RQ3. Information controls come first because they identify whether learned output helps the final solver. The comparison with argmax plus descent then asks whether richer decoding adds value. Cross-optimizer and family analyses establish how broadly those conclusions apply, followed by decoder variants and computational cost.

| Priority | Subquestion | Role in the argument | Results section |
| ---: | --- | --- | --- |
| 1 | Does trained $P$ improve the final result over uninformative, scrambled, shuffled, or untrained $P$ under the same decoder? | Separates useful learned coordination from the decoder's own graph-based search | 4.1 |
| 2 | Does D-CE extract a substantial benefit beyond argmax followed by identical descent? | Tests whether richer decoding adds value after elementary repair is accounted for | 4.2 |
| 3 | Does applying the same stronger decoder erase differences between optimization methods? | Tests whether decoding determines the method comparison | 4.3 |
| 4 | Which graph families benefit from learned information? | Resolves the heterogeneous aggregate comparison with the graph-only control | 4.4 |
| 5 | Do saturation order or repeated sampling improve the trained-output result? | Checks whether order and sampling explain the decoder effect | 4.5 |
| 6 | Are the observed gains justified by decoder and repair cost? | Distinguishes a useful inexpensive decoder from costly repeated search | 4.6 |

These priorities organize the report; they do not retrospectively change the protocol's primary and secondary comparison labels. All analyses reuse existing training runs and require no retraining.

## 2 Objective and mathematical foundation

### 2.1 Shared relaxation, conflicts, and repair

For a simple undirected graph $G=(V,E)$, a soft coloring has one probability row $p_v\in\Delta_k$ per vertex. With row-wise softmax, $P=\operatorname{softmax}(Z)$, the shared Potts objective and hard conflict count are

$$
L(P)=\sum_{\{u,v\}\in E}p_u^\top p_v,
\qquad
C(c)=\sum_{\{u,v\}\in E}\mathbf1[c_u=c_v].
$$

Independent sampling, $c_v\sim p_v$, gives $\mathbb E[C(c)]=L(P)$. Argmax instead selects the highest-probability color at each vertex and has no corresponding guarantee that $C(c)\leq L(P)$.

Every decoded coloring also receives the same steepest strict vertex descent used in RQ1. If $M_v(a)$ counts neighbors currently assigned color $a$, an improving recoloring exists at $v$ when

$$
M_v(c_v)-\min_a M_v(a)>0.
$$

Descent repeatedly chooses a largest strictly improving move, with seeded random tie-breaking, and stops when none remains. It excludes neutral and worsening moves. We report both raw conflicts and conflicts after descent, denoted $C_{\mathrm{desc}}$. The latter measures the quality of the reached single-flip local minimum, not a gap to the unknown global optimum.

The RQ1 probability-space lemma and softmax-saturation analysis remain relevant background, but their proofs and diagnostics are not repeated here. RQ3 intervenes on the decoder and its probability input rather than on training dynamics.

### 2.2 Conditional-expectation decoding

D-CE fixes vertices one at a time. Already fixed neighbors contribute one-hot rows; unfixed neighbors retain their original probability rows. For an unfixed vertex $v$, define the expected incident conflicts from choosing color $a$ as

$$
S_v(a)=\sum_{u\in N(v)\cap F}\mathbf1[c_u=a]
       +\sum_{u\in N(v)\setminus F}p_u(a),
$$

where $F$ is the set of fixed vertices. D-CE chooses a color minimizing $S_v(a)$. Before fixing $v$, the conditional expectation averages these costs using $p_v$, so its minimum cannot exceed that average. Each fixation therefore does not increase the conditional expected total conflicts. In exact arithmetic, after all vertices are fixed,

$$
C(\operatorname{DCE}(P))\leq L(P).
$$

Subsequent strict descent can only reduce conflicts further. The implementation treats color costs within $10^{-9}$ of the minimum as tied to avoid splitting numerical ties; the bound is checked with numerical tolerance.

The main order selects the vertex with the most distinct colors among fixed neighbors, then largest degree, then lowest vertex index. Color ties use the lowest index. The random-order control uses one seeded random permutation while retaining the same color-selection rule. The implemented D-CE has cost $O(n^2+mk)$, with $n=|V|$ and $m=|E|$.

The guarantee compares D-CE with $L(P)$, not with argmax plus descent. It does not ensure that the local minimum reached from D-CE is better than the one reached from argmax. When $P$ is one-hot, D-CE becomes a sequential least-conflict recoloring sweep of the evolving hard assignment. It may change tied colors, including through neutral recolorings, rather than copying the assignment. Thus, even a D-CE improvement from a nearly hard output need not come from information in its small alternative-color probabilities.

### 2.3 Exactly uniform input and the implemented graph-only control

For exactly uniform rows, $p_v(a)=1/k$, the unfixed-neighbor term in $S_v(a)$ is independent of color. D-CE then chooses a least-conflicting color among fixed neighbors in DSATUR's vertex order. With matching deterministic tie rules, it is exactly fixed-$k$ DSATUR; the repository tests this identity.

The experimental variant named `uniform` is instead

$$
P^{\mathrm{null}}=\operatorname{softmax}(0.1\,\Xi),
\qquad \Xi_{v,a}\overset{\mathrm{iid}}\sim N(0,1).
$$

It is near-uniform, independently generated, and contains no trained graph-specific probabilities. D-CE still uses the graph. The perturbations affect expected color costs, so this control is DSATUR-like rather than an exact rerun of deterministic DSATUR. We call it **near-uniform D-CE** below. The exact identity at uniform input explains its role, while the distinction prevents identifying its numerical results with every DSATUR variant studied in RQ1.

## 3 Shared experimental setup

### 3.1 Graphs, frozen outputs, and reused components

We use the same 68 DIMACS graphs, recorded proven chromatic numbers $\chi\geq3$, and seeds 0–4 as RQ1 and RQ2. Graphs have 11–1,000 vertices, and every run uses $k=\chi-1$. Assuming the recorded chromatic numbers and graph files are correct, zero conflicts are impossible. The chromatic number does not specify the minimum positive conflict count at this palette size.

The PI-GNN reference, arm A, is the frozen two-layer PI-SAGE configuration described in RQ1: trainable vertex embeddings, mean-neighbor aggregation, hidden dropout, and per-instance AdamW training. RQ3 loads each saved evaluation-logit checkpoint at exactly 100,000 steps and recomputes softmax in float64. It does not choose a best earlier checkpoint, use the selected 300,000-step extensions, or retrain for any decoder.

We reuse the graph manifest, conflict counter, single-flip test, and strict descent implementation. The 340 trained-A argmax outputs reproduce the corresponding raw and repaired RQ1 conflict counts exactly. Detailed architecture, tuning searches, saturation diagnostics, and matched-training-time results remain in the companion papers; only their consequences for interpreting decoding are repeated here.

### 3.2 Probability-input controls

Every A graph–seed run is decoded under all five inputs in Table 1.

| Input | Construction | What the comparison tests |
| --- | --- | --- |
| Trained | The run's saved 100k evaluation probabilities | Reference learned assignment |
| Near-uniform (`uniform`) | Softmax of independent Gaussian logits with standard deviation 0.1 | What graph-aware decoding achieves without trained probabilities |
| Untrained | Softmax of A's saved initial evaluation logits `z0.pt` | What training adds beyond graph-dependent random network output |
| Vertex-shuffled (`rowperm`) | One seeded permutation of trained rows across vertices | Whether assigning the learned rows to their correct vertices matters |
| Color-scrambled (`colperm`) | An independent permutation of color coordinates within every trained row | Whether coordination of color identities matters while preserving each vertex's confidence and entropy |

*Table 1. Frozen probability inputs. Vertex shuffling preserves the multiset of rows but changes their graph alignment; color scrambling preserves every row's probability values but disrupts cross-vertex coordination.*

A single common permutation of all color columns would leave coloring conflicts unchanged and would not test coordination. The implemented color permutations are independent across rows. These are finite seeded perturbation controls, with one realization per graph and seed, rather than exhaustive averages over all possible permutations.

Degenerate inputs are retained. Four A runs on DSJR500.5 have every edge in conflict under trained argmax. This flag identifies an extreme hard-output failure; equal argmax colors alone do not establish that their full probability rows are identical or that row permutation must be a no-op.

### 3.3 Decoders and additional optimization arms

Each input receives argmax, saturation-order D-CE, random-order D-CE, and $K=32$ independent sampled colorings. All outputs are evaluated before and after strict descent. Sampling reports both mean and best counts; its headline comparison uses the best raw sample and the best repaired sample separately. They need not be the same sample. Each sampled coloring receives its own descent run, so best-of-32 is a repeated-search procedure with a larger repair budget.

Control generation, sampling, and random vertex order use separate seeded random-number streams. Descent uses `default_rng(seed)` for every decoded start. Sharing the descent seed makes the procedure consistent; different starts can still follow different tie sequences and reach different local minima.

The same decoders are applied to the eight additional RQ2 arms at 100k: A without dropout; plain free logits B from A's initial output; scale-matched random-start free logits B-rand (`Brand`); free logits with training-logit noise $\sigma=0.5,1,2$; the node MLP C without message passing; and the initialization-linearized PI-GNN D. The full arm definitions and separate tuning of A, B, C, and D are in RQ2. Noisy B reuses B's optimizer settings; its noise strengths were not independently tuned on the synthetic set.

These eight arms receive trained and color-scrambled inputs. Their near-uniform reference is reused from A because its construction depends on graph size, palette, and seed rather than the optimization arm. Overall, the study contains 3,060 graph–seed–arm run files and 28,560 input–decoder records: 6,800 for A and 2,720 for each other arm. A sampling record summarizes 32 decoded and repaired colorings.

### 3.4 Metrics, aggregation, and interpretation rules

The primary metric is conflicts after descent, with raw conflicts reported alongside. For a first method or input $X$ and reference $Y$, we average five seeds within each graph and compute

$$
d_G=\frac{\overline C_X(G)-\overline C_Y(G)}{\max(\overline C_Y(G),1)}.
$$

Negative differences favor the first method. A practical tie occurs when the absolute seed-mean difference is below one conflict **or** $|d_G|\leq0.05$. Other cases are wins or losses. Graphs receive equal weight in the reported conflict means and medians, although a few graphs with large counts can still dominate an arithmetic mean.

The analysis labels three comparisons primary: trained D-CE versus trained argmax, trained versus near-uniform D-CE, and trained versus color-scrambled D-CE. Other nulls, order, sampling, arm, and family comparisons are secondary. All p-values are descriptive two-sided Wilcoxon signed-rank values on nonzero graph-level relative differences. The RQ3 analysis uses 0.05 as an operational threshold without a multiplicity adjustment, and no confidence intervals are supplied.

A directional reading additionally requires a median difference beyond the 5% band and more practical wins than losses. The stored phrase “not distinguishable” therefore covers both a nonsignificant test and a significant but practically small median. We state which applies. Neither case establishes equivalence. The qualitative interpretation table was written before the main results; the Experiment 1 practical tie rule was adopted unchanged after an initial strict-tie look at A, as disclosed in the protocol log.

## 4 RQ3 experiments and analyses

### 4.1 Trained probabilities beat corrupted information without a broad advantage over the graph-only control

**Question and purpose.** Does the learned probability matrix improve the final result when every input receives the same graph-aware decoder and repair? This is the central information test: it distinguishes learned coordination from what D-CE can construct using the graph itself.

**Procedure.** Apply saturation-order D-CE and then identical strict descent to A's trained, near-uniform, untrained, vertex-shuffled, and color-scrambled probabilities. Average seeds within each graph and compare the resulting endpoints. We also compare near-uniform with color-scrambled probabilities to check whether scrambling is a neutral reference or actively misleading.

**Results.** Table 2 reports the first input relative to the second; every row uses D-CE followed by descent.

| Comparison | Better / tie / worse | Median relative difference | Mean conflicts, first / second | Descriptive Wilcoxon p |
| --- | --- | ---: | ---: | ---: |
| Trained vs near-uniform | 27 / 28 / 13 | 0.0% | 111.36 / 121.12 | 0.227 |
| Trained vs color-scrambled | 48 / 20 / 0 | −25.5% | 111.36 / 126.97 | $2.40\times10^{-12}$ |
| Trained vs vertex-shuffled | 56 / 12 / 0 | −29.9% | 111.36 / 128.52 | $1.64\times10^{-12}$ |
| Trained vs untrained | 51 / 14 / 3 | −22.2% | 111.36 / 127.63 | $1.36\times10^{-8}$ |
| Near-uniform vs color-scrambled | 41 / 24 / 3 | −17.5% | 121.12 / 126.97 | $4.78\times10^{-11}$ |

*Table 2. Probability-information controls on all 68 graphs. The first two rows are primary comparisons; the others are secondary. Negative differences favor the first input.*

Training produces useful structure relative to the corrupted controls. In particular, retaining each vertex's confidence and entropy while scrambling its color coordinates worsens the repaired result on 48 graphs and improves none under the practical rule. Preserving the entire row multiset while assigning rows to different vertices is also harmful. These comparisons support useful color coordination and vertex alignment; they do not support confidence alone as the useful property.

However, trained A has zero median difference from near-uniform D-CE and a descriptive $p=0.227$. Its lower arithmetic mean and more wins than losses do not satisfy the operational rule for a broad directional advantage. The family analysis in Section 4.4 explains why this aggregate result should not be described as uniformly absent information.

The last row changes the interpretation of the scrambled control. A decoder with uninformative probabilities outperforms the same decoder supplied with confident but miscoordinated probabilities. Therefore, beating scrambled $P$ cannot by itself establish an advantage over graph-only decoding. The plan's grouped phrase “null or permuted” hides this difference: the uninformative and corrupted inputs give different answers and must remain separate.

For comparison with RQ1, trained argmax plus descent beats near-uniform argmax plus descent on 60 graphs, ties on seven, and loses on one, with median $d=-40.6\%$. Thus training is useful relative to a random hard start for elementary descent. Replacing the null input's argmax with graph-aware D-CE raises the reference substantially and removes the established broad trained-input advantage. The information contribution depends on the discrete procedure receiving the input.

### 4.2 D-CE's large raw improvement becomes modest after elementary repair

**Question and purpose.** Does D-CE recover a substantial benefit that argmax followed by strict descent misses? Comparing only raw colorings would mix useful probability decoding with work that a simple discrete improvement procedure can already perform.

**Procedure.** Decode each trained A probability matrix with argmax and saturation-order D-CE. Compare both raw outputs and their endpoints after identical descent. This changes only decoding; the graph, saved probabilities, and training budget are fixed.

| Metric | D-CE better / tie / worse than argmax | Median relative difference | Mean D-CE conflicts | Mean argmax conflicts | Descriptive Wilcoxon p |
| --- | --- | ---: | ---: | ---: | ---: |
| Raw | 33 / 35 / 0 | −11.4% | 125.14 | 1,517.35 | $1.63\times10^{-10}$ |
| After identical descent | 22 / 46 / 0 | −2.8% | 111.36 | 114.86 | $1.12\times10^{-6}$ |

*Table 3. Decoder gain on frozen trained PI-GNN outputs. The very large raw mean difference is strongly affected by high-conflict failures on a few graphs; the median is computed from individual graph-level relative differences.*

The raw improvement is substantial, but most of its mean-conflict reduction can also be obtained by descent from argmax. Once that repair is included, the median advantage falls inside the 5% practical band. This is a small but consistent improvement rather than exact equality: the signed-rank value is small, 22 graphs improve beyond the practical thresholds, and none worsens beyond them.

The aggregate description has meaningful exceptions. D-CE improves repaired A on all three fpsol and all three inithx graphs, with family median reductions of 42.3% and 41.9%, respectively; on DSJR it wins twice and ties once, with median −31.9%. These improvements still do not make trained D-CE superior to the graph-only reference on those families. Queen graphs show the complementary pattern: D-CE and argmax plus descent practically tie on 11 of 12, yet trained D-CE beats the near-uniform reference on all 12. Learned coordination and extra decoder gain are separate properties.

The implemented comparison does not isolate information lost by argmax in a strict causal sense. It changes both access to soft probabilities and the discrete update procedure. D-CE on an exactly hardened version of the same trained $P$ was not included as a separate arm. Since D-CE can improve hard assignments through its recoloring sequence, the modest residual gain cannot all be attributed to alternative-color probability mass. Conversely, its small size does not prove that no useful residual soft information exists.

### 4.3 Stronger decoding reduces some optimization differences but does not erase them

**Question and purpose.** Do the optimization methods studied in RQ2 mainly differ because argmax is an inadequate decoder? If so, applying the same graph-aware decoder should make their repaired results converge.

**Procedure.** Apply saturation-order D-CE and strict descent to every arm's trained 100k probabilities. Compare each with its own argmax-plus-descent output, with A under the same decoder, and with the common near-uniform D-CE reference. The training conditions are inherited from RQ2; this is a common-step decoder study, not a new matched-total-time comparison.

| Arm | Mean argmax + descent | Mean D-CE + descent | Mean-conflict rank, argmax → D-CE | D-CE better / tie / worse than own argmax + descent | Median decoder difference |
| --- | ---: | ---: | --- | --- | ---: |
| B-noise-2 | 83.41 | 83.39 | 1 → 1 | 0 / 68 / 0 | 0.0% |
| B-noise-1 | 92.47 | 92.32 | 2 → 2 | 1 / 67 / 0 | 0.0% |
| B-rand | 107.07 | 106.22 | 3 → 3 | 5 / 63 / 0 | −1.4% |
| B, plain free logits | 108.61 | 107.34 | 4 → 4 | 5 / 63 / 0 | −0.8% |
| B-noise-0.5 | 109.21 | 108.35 | 5 → 5 | 4 / 64 / 0 | −0.4% |
| A, PI-GNN | 114.86 | 111.36 | 6 → 6 | 22 / 46 / 0 | −2.8% |
| A-no-dropout | 115.72 | 117.02 | 7 → 8 | 10 / 48 / 10 | −0.5% |
| C, node MLP | 120.41 | 116.15 | 8 → 7 | 27 / 41 / 0 | −6.2% |
| D, fixed neural linearization | 130.45 | 123.08 | 9 → 9 | 45 / 23 / 0 | −14.6% |

*Table 4. Common-decoder results on all nine arms, with five seeds on every graph. Means give equal weight to graphs. Mean-conflict ranks are descriptive and need not agree with paired graph-level verdicts.*

D-CE adds little to the direct-logit arms: 63–68 graphs practically tie for each. It helps C and D more, satisfying the directional rule for those within-arm decoder comparisons. D nevertheless remains last by mean conflicts. For A-no-dropout, D-CE produces ten practical wins and ten losses relative to its own repaired argmax and increases the mean conflict count. This illustrates why the conditional-expectation guarantee is not a guarantee against argmax followed by descent.

The broad mean ordering persists, with one swap between C and A-no-dropout. More importantly, the paired comparisons retain meaningful optimizer differences. After D-CE, A beats plain B on 30 graphs, ties on 24, and loses on 14; the median difference for B relative to A is +7.6%, with descriptive $p=0.000347$. A retains this paired advantage despite B having the lower arithmetic mean.

B-noise-1 and B-noise-2 still outperform A under the operational rule. Their respective win/tie/loss counts are 31/30/7 and 35/29/4, with median differences of −12.9% and −18.7%. D loses to A on 46 graphs, ties on 21, and wins on one, with median difference +20.0%. Thus the stronger decoder does not make the optimizer comparisons disappear.

Some RQ2 gaps do narrow enough to change their operational reading. For C versus A, median $d$ falls from +7.8% after argmax plus descent to +4.7% after D-CE plus descent; B-noise-0.5 versus A similarly falls from +5.8% to +4.1%. Both D-CE comparisons retain small descriptive p-values but fall inside the practical median band. It would therefore overstate the results to say that every method comparison is unchanged.

For a graph-level measure across all nine arms, define

$$
R_G=\frac{\max_a\overline C_a(G)-\min_a\overline C_a(G)}
           {\max(\operatorname{mean}_a\overline C_a(G),1)}.
$$

The median $R_G$ decreases from 0.865 after argmax plus descent to 0.685 after D-CE plus descent. Comparing spreads within graphs gives 44 reductions, 16 practical ties, and eight increases, using a 5% relative tie band without the one-conflict clause. The median within-graph relative reduction is 10.2%, with stored descriptive $p\approx2\times10^{-7}$. It differs from the reduction between the two marginal medians. Decoding narrows differences but leaves substantial optimizer dependence.

Every arm's trained input outperforms its own color-scrambled input under the operational rule: 39–58 practical wins, with zero losses for seven arms, one for D, and two for A-no-dropout. Useful coordination is therefore not unique to message passing. Against near-uniform D-CE, B-noise-2 is the only tested arm satisfying the directional improvement rule: 36 wins, 27 ties, five losses, median −21.2%, and $p=0.000741$. B-noise-1 has 36 wins, 25 ties, and seven losses, but its $p=0.0819$ does not pass that rule. These are descriptive comparisons with the near-uniform reference, not new claims of superiority over all DSATUR variants or TabuCol.

### 4.4 The value of trained information depends strongly on graph family

**Question and purpose.** Does the aggregate lack of a directional advantage over near-uniform D-CE describe every graph family? Family stratification determines where the learned input helps or misleads the same decoder.

**Procedure and metrics.** Use the trained-A versus near-uniform comparison from Section 4.1, after saturation-order D-CE and strict descent, and group the graph-level verdicts by the manifest's families. Table 5 preserves individual families, including the register-allocation subfamilies. The small groups are descriptive strata rather than separate confirmatory experiments.

| Family | Graphs | Trained better / tie / worse | Median relative difference | Mean trained / near-uniform conflicts |
| --- | ---: | --- | ---: | ---: |
| Queen | 12 | 12 / 0 / 0 | −18.5% | 29.53 / 35.65 |
| DSJC | 4 | 3 / 1 / 0 | −16.1% | 32.70 / 39.05 |
| School | 2 | 2 / 0 / 0 | −13.0% | 350.50 / 400.50 |
| Flat | 5 | 3 / 2 / 0 | −5.1% | 644.32 / 668.96 |
| Leighton | 12 | 7 / 0 / 5 | −7.1% | 255.05 / 286.13 |
| DSJR | 3 | 0 / 1 / 2 | +44.0% | 17.93 / 12.33 |
| fpsol | 3 | 0 / 2 / 1 | +57.1% | 2.13 / 1.13 |
| inithx | 3 | 0 / 1 / 2 | +140.0% | 2.60 / 1.00 |
| Miles | 5 | 0 / 2 / 3 | +100.0% | 2.32 / 1.08 |
| zeroin | 3 | 0 / 3 / 0 | +20.0% | 1.20 / 1.00 |
| mulsol | 5 | 0 / 5 / 0 | 0.0% | 1.68 / 1.60 |
| Mycielski | 5 | 0 / 5 / 0 | 0.0% | 1.04 / 1.08 |
| Social networks | 5 | 0 / 5 / 0 | 0.0% | 1.12 / 1.08 |
| Other, games120 | 1 | 0 / 1 / 0 | 0.0% | 2.00 / 2.00 |

*Table 5. Family dependence of learned-input value under the same decoder. Counts sum to 27 trained wins, 28 ties, and 13 losses. Positive differences favor the near-uniform control.*

Queen graphs provide the clearest family-level evidence that trained probabilities supply useful information beyond the near-uniform reference. DSJC, school, and flat also show favorable practical verdicts, while Leighton is mixed. DSJR and parts of register allocation and miles favor the graph-only control. This qualitatively agrees with the classical-start heterogeneity in RQ1 without identifying the near-uniform control with its exact DSATUR implementations.

Large percentage differences on low-conflict families must be read alongside absolute counts. For example, zeroin has a +20% median difference but three practical ties because every absolute seed-mean difference is below one conflict. Conversely, the queen gains are broadly shared across its 12 graphs. The overall zero median is therefore a heterogeneous benchmark result, not evidence that trained information is equally useless everywhere.

### 4.5 Random order and best-of-32 sampling add little for trained PI-GNN

**Question and purpose.** Is D-CE's benefit primarily due to saturation order, and can repeated sampling exploit trained probabilities that argmax misses? These controls test two alternative routes from the same $P$ to a hard start.

**Procedure.** Compare saturation-order D-CE with a seeded random vertex permutation using the same expected-conflict color rule. Separately, sample 32 independent colorings from each trained A matrix, repair all 32, and retain the smallest repaired conflict count. The sampling comparison receives more local-search attempts than argmax and is not an equal-cost comparison.

| Comparison after descent | First better / tie / worse | Median relative difference | Mean conflicts, first / second | Descriptive Wilcoxon p |
| --- | --- | ---: | ---: | ---: |
| Saturation-order vs random-order D-CE | 4 / 64 / 0 | 0.0% | 111.36 / 112.84 | 0.162 |
| Best of 32 repaired samples vs repaired argmax | 2 / 65 / 1 | 0.0% | 113.34 / 114.86 | 0.00806 |

*Table 6. Order and sampling controls on trained A. These are secondary comparisons. The sampling p-value includes small nonzero differences classified as practical ties.*

Saturation order has no established overall directional advantage for these trained inputs, although four graphs improve beyond the practical thresholds. The result concerns this fixed $P$, one random order per seed, and subsequent descent; it does not establish that vertex order is irrelevant for uninformative inputs or other decoding problems.

Sampling similarly gives little practical improvement despite 32 repair attempts. It ties on 65 graphs, and its zero median fails the directional rule even though the descriptive signed-rank value is small. It can lose because the finite sample set need not contain argmax or a coloring reaching the same local minimum.

The RQ1 confidence results help interpret this outcome: most trained probability rows are already highly concentrated, so sampling often provides limited diversity. Exactly one-hot rows reproduce argmax deterministically under sampling. This is a mathematical limiting case and a plausible explanation, rather than a measured attribution of every sampling tie. No temperature sweep, larger sampling budget, or restart schedule was tested here.

### 4.6 D-CE is inexpensive in this implementation; repeated sampling dominates cost

**Question and purpose.** Does the modest repaired-output gain require enough extra work to undermine the decoder comparison? The plan's possible “slightly better but much slower” interpretation must be checked against measured decoder and repair time.

**Procedure and timing scope.** Sum the recorded decoding and descent seconds across A's 340 runs. Table 7 separates the trained input from the full five-input control study. These are elapsed operation times for CPU tasks, excluding neural training, checkpoint loading, and other task overhead. They are not direct measurements of process CPU utilization. Sampling includes descent on all 32 samples.

| Decoder plus strict descent | Trained input: decode seconds | Trained input: descent seconds | Trained input: total seconds | All five A inputs: total seconds |
| --- | ---: | ---: | ---: | ---: |
| Argmax | 0.44 | 29.94 | 30.39 | 348.58 |
| Saturation-order D-CE | 11.49 | 9.83 | 21.32 | 140.66 |
| Random-order D-CE | 6.17 | 8.73 | 14.90 | 115.26 |
| Best-of-32 sampling procedure | 0.54 | 788.80 | 789.34 | 10,475.05 |

*Table 7. Sums of logged operation times, recomputed from A's decoder records. Each trained-input column covers the same 340 graph–seed runs; the final column covers 1,700 input instances.*

D-CE has higher decoding overhead than argmax, but its starts require less repair. For trained A, D-CE plus descent takes 21.32 summed seconds versus 30.39 for argmax plus descent. Thus the specific interpretation that D-CE is only slightly better but much slower is unsupported by these measurements. Its modest quality gain can still be useful as an inexpensive implementation choice; it does not establish a major information advantage.

Summing complete A task durations gives 3.22 task-hours, with median 6.56 seconds and maximum 883.33 seconds per run. Sampling and its 32 descents account for 2.91 hours, about 90.4% of that sum. The eight additional arms together require 6.94 summed task-hours, each with two probability inputs rather than A's five. These arm totals are not directly comparable measures of training or optimizer efficiency.

The experiment log reports approximately 19 minutes of cluster elapsed time for the main A batch and 30 minutes for the other arms under concurrent scheduling. Shared-node contention affects these totals and the slowest tasks. They should not be treated as isolated-machine benchmarks. The cost evidence favors D-CE over the tested sampling procedure for these outputs, while a matched-total-budget comparison of complete solvers remains outside this study.

## 5 Discussion

RQ3 has a conditional answer. The learned PI-GNN output contains useful vertex alignment and color coordination relative to untrained or deliberately corrupted probabilities. However, those controls are not interchangeable with an uninformative input: confidently miscoordinated rows can impede D-CE, while a near-uniform input allows it to construct a competitive coloring directly from the graph. The completed controls support useful learned structure without establishing a broad practical advantage over the graph-only reference.

The large difference between raw D-CE and raw argmax also has to be interpreted through repair. Once both outputs receive elementary strict descent, the aggregate D-CE advantage is modest. This supports the plan's interpretation that simple discrete improvement explains much of the apparent decoder gain. It does not make D-CE unnecessary: it remains inexpensive here and provides useful improvements on particular families. Nor does it identify the residual improvement as purely soft information, because the implemented interventions change both input use and the discrete recoloring sequence.

The cross-arm results constrain a decoder-dominance explanation. D-CE reduces the spread and changes some practical comparisons, but it does not remove the advantage of sufficiently noisy direct logits over A or rescue the fixed linearization. The procedure producing $P$ remains consequential after common graph-aware decoding. Since every optimizer uses the graph in the Potts objective, useful graph-specific probabilities do not imply that the forward model needs message passing. RQ2's noise, architecture, and fixed-linearization analyses provide that separate test.

Family dependence is part of the answer rather than an exception to hide. Trained probabilities help near-uniform D-CE on every queen graph, while the graph-only reference is stronger on parts of DSJR, register allocation, and miles. A single aggregate mean, zero median, or operational “not distinguishable” label cannot express this structure. The appropriate conclusion is limited aggregate additional value with identifiable family-specific benefits and failures.

The stronger-search context from RQ1 also remains relevant. Random-start TabuCol found fewer conflicts than repaired A on 41 graphs, with 13 ties and 14 A wins, subject to the documented hardware, initialization, seed-count, and temporal-selection differences. That reference separates a useful local start from strong overall coloring performance. RQ3 does not provide a new time-matched TabuCol comparison with D-CE or noisy free logits, and its improvements should not be interpreted as establishing one.

## 6 What this achieves in the general project plan

The plan's Experiment 2 answers RQ3; Experiment 1 answers RQ1, and Experiment 3 answers RQ2. The frozen-output decoder study and its extension across all completed optimization arms are complete at the common 100k budget. Earlier companion-paper statements describing decoder work as remaining are historical; the completed RQ3 evidence is reported here and in the Experiment 2 log.

| Planned component | Achievement supported by this repository | Status and remaining scope |
| --- | --- | --- |
| Same-output decoder comparison | Argmax, saturation-order D-CE, random-order D-CE, and 32-sample decoding on frozen 100k probabilities | Completed for 68 graphs × 5 seeds; no decoder-specific retraining |
| Separate decoding from local repair | Raw and repaired conflicts recorded for every input and decoder; each start receives the same strict descent | Completed; D-CE's large raw gain becomes modest overall after common repair |
| Null and permuted probability controls | Near-uniform, untrained, vertex-shuffled, and independently color-scrambled A inputs | Completed; trained beats corrupted inputs without an established broad advantage over near-uniform D-CE |
| Vertex-order control | Saturation order compared with seeded random order | Completed; little overall additional benefit on trained A after repair |
| Sampling and repair | Mean and best results from 32 independent colorings, all repaired | Completed; headline best-of-32 gain is practically small and expensive |
| End-to-end RQ2 decoder view | All nine optimization arms decoded at the same 100k checkpoint; trained and color-scrambled inputs for the eight additional arms | Completed; some differences narrow, but optimizer dependence persists |
| Family stratification | Primary comparisons broken down by the manifest's 14 families | Completed as descriptive evidence; no independent large sample for small families |
| Runtime accounting | Separate decoder and descent times, plus complete task durations | Completed for post-processing; no matched-total-budget full-solver ranking |
| Paired statistics and uncertainty | Graph-level practical verdicts, median differences, arithmetic means, and descriptive signed-rank values | Available; confidence intervals proposed in the general plan were not computed |
| Broader endpoint and mechanism studies | RQ1 endpoint results and RQ2 optimization controls provide the shared context | Core three-experiment decomposition is available; this does not claim completion of every optional spectral, architecture, warm-start, or mean-field extension |

The plan's strongest information claim requires substantial improvement over argmax plus repair and over both uninformative and corrupted inputs. The aggregate PI-GNN results do not satisfy that combination. Its alternative claim that trained $P$ is no more informative than a null **or permuted** input is also too broad: the corrupted-input comparisons clearly favor training. Likewise, common decoding does not erase optimizer differences. The completed outcome is a more specific decomposition: learned coordination helps relative to damaged inputs, graph-only decoding remains competitive overall, and elementary repair accounts for most of the large raw decoder improvement.

## 7 Limitations

The study concerns one frozen PI-SAGE family, the implemented RQ2 arms, a heterogeneous 68-graph benchmark, and the infeasible regime $k=\chi-1$. It does not establish a ranking for all GNNs, decoders, graph distributions, or feasible palettes. Chromatic-number claims are inherited from dataset curation. Only the 100k checkpoint is decoded here; selected longer-budget RQ1 outputs do not establish decoder robustness at 300k.

The `uniform` control is near-uniform rather than exactly uniform. Its D-CE result is a graph-only, DSATUR-like reference, not numerically identical to deterministic DSATUR or its randomized variants. Permutations can create misleading inputs, so a trained-input advantage over them is insufficient evidence of value beyond a graph-only solver. Five perturbation realizations per graph also do not characterize every possible corruption or order.

D-CE changes the discrete procedure as well as access to probabilities. Without a separate D-CE comparison on hardened trained rows, the observed gain cannot be uniquely attributed to non-argmax probability information. The conditional-expectation guarantee concerns $L(P)$, not the repaired endpoint from argmax; neutral choices and different basins can change the eventual local minimum. Finite-budget negative results do not prove that another decoder could not extract more useful information.

Sampling uses one fixed budget of 32 and supplies 32 repair attempts. It has no matched-cost counterpart, temperature search, or restart study. Concentration provides a plausible explanation for limited diversity, but the completed report does not measure the number of distinct sampled assignments or causally attribute every tie to saturation.

The optimizer-arm extension inherits the tuning and intervention limits of RQ2. In particular, the noisy-logit strengths were not separately tuned on the disjoint synthetic set, and selecting the best observed strength on DIMACS is descriptive. C changes several architectural and tuned settings together; A-no-dropout is an intervention on A's frozen configuration rather than an independently retuned model. RQ3 uses equal training steps and common post-processing rather than equal total time or compute.

All inferential summaries are descriptive benchmark evidence. The practical rule was adopted after an initial look using strict ties, p-values are unadjusted in the RQ3 analysis, and no confidence intervals or equivalence tests were computed. Wilcoxon tests retain small nonzero differences inside the practical tie band. Large percentage changes can represent less than one conflict, and large-count graphs can influence arithmetic means despite graph-balanced aggregation. Family results have unequal and sometimes very small sample sizes.

Timing records are elapsed durations from shared cluster CPU tasks. Sums across tasks are useful accounting figures rather than direct CPU-utilization or isolated-runtime measurements. They exclude the training cost already incurred to generate $P$, and cluster batch elapsed time depends on concurrency. No new matched-budget comparison against stronger local search is established here.

## 8 Conclusion

Trained PI-GNN probabilities retain useful vertex and color coordination relative to shuffled, scrambled, and untrained outputs, but their D-CE result has no established overall advantage over near-uniform graph-only decoding. D-CE's large improvement over raw argmax becomes modest after both outputs receive strict vertex descent, with larger benefits on selected families. Random order and 32-sample search add little for trained A, while the latter dominates measured post-processing cost. Applying common D-CE narrows some optimization differences but preserves the main advantage of noisy direct logits and the shortfall of the fixed linearization. RQ3 therefore supports a joint account: the optimizer determines a consequential starting structure, elementary discrete repair explains much of the apparent decoder gain, and learned soft output supplies conditional rather than universal additional value.

## References and evidence

1. [Mechanistic experiment plan](PI_GNN_Mechanistic_Plan_Final_v3.pdf), especially the shared definitions, Experiment 2, and its interpretation table. The implemented 68-graph set expands the plan's earlier proposed set.
2. [RQ1 paper](RQ1_Paper.md). Shared PI-GNN architecture and tuning, probability-space endpoint foundation, confidence diagnostics, strict descent, and classical quality references.
3. [RQ2 paper](RQ2_Paper.md). Same-start free logits, initialization, noise, node-MLP, fixed-linearization, and training-budget controls.
4. [Experiment 2 implementation and result log](Experiment2_Steps.md). Decoder protocol, control construction, run history, analysis-rule disclosure, runtime, and completed RQ3 results.

The contribution is the controlled comparison and its information controls, rather than a claim that conditional-expectation rounding, DSATUR, or vertex descent is new. Background literature and the original PI-GNN formulation are documented in the companion papers and general plan.

| Evidence | Repository location |
| --- | --- |
| Graphs, palettes, and family metadata | [Sweep manifest](../configs/sweep_manifest.csv), [chromatic-number bounds](../data/dimacs/chi_bounds.csv) |
| Frozen PI-GNN configuration and architecture | [A configuration](../configs/pignn_frozen.json), [model implementation](../src/graphsml/model.py), [RQ1](RQ1_Paper.md) |
| D-CE, sampling, null inputs, and numerical conventions | [Decoder implementation](../src/graphsml/decoders.py), [decoder tests](../tests/test_decoders.py) |
| Graph-only identity and strict local repair | [DSATUR implementation](../src/graphsml/controls.py), [endpoint implementation](../src/graphsml/endpoint.py), [endpoint tests](../tests/test_endpoint.py) |
| Frozen-checkpoint loading, random streams, and sampling selection | [Decoder driver](../scripts/run_decoders.py), [cluster wrapper](../scripts/run_decoders.sbatch) |
| Tables 2, 3, and 6, and raw comparisons | [PI-GNN comparison summary](../results/analysis/exp2_summary.csv) |
| Table 4 and cross-optimizer comparisons | [Arm comparison summary](../results/analysis/exp2_arms.csv), [RQ2 arm definitions](RQ2_Paper.md) |
| Between-arm spread | [Graph-level spread](../results/analysis/exp2_arm_spread.csv), [comparison implementation](../scripts/analyze_exp2.py) |
| Table 5 and decoder effects by family | [Family comparisons](../results/analysis/exp2_by_family.csv) |
| Table 7 and completion/timing audit | Per-run `decoders.jsonl` and `DONE` records in [decoder results](../results/exp2/); [task-cost summary](../results/analysis/exp2_cost.csv) |
| Reproduction of RQ1 argmax endpoints | Per-run `endpoint_trajectory.jsonl` in [A training results](../results/main/) |
| Completed runs and historical decisions | [Experiment 2 log](Experiment2_Steps.md), [RQ2 progress log](progress_exp3.md) |

For this paper's read-only audit, each of the nine arms has exactly the expected 68 graphs × five seeds and 340 completion records. The 28,560 decoder records have unique graph–seed–arm–input–decoder keys, the expected input and decoder coverage, epoch 100,000, and graph dimensions and palettes matching the manifest. Every logged repaired count is no greater than its corresponding raw count, including best-of-sample summaries, and none is zero. Every D-CE raw count satisfies the logged Potts-loss bound within $10^{-7}$. All 340 trained-A argmax raw and repaired counts match the saved RQ1 endpoints. Recomputed graph-level verdicts, medians, and means agree with the stored PI-GNN, arm, and family comparison CSVs; p-values are taken from the recorded analysis. Table 7 is recomputed from the logged operation times. The corrected final spread analysis is used, with a relative-only tie rule for dimensionless spreads, as documented in the Experiment 2 log.
