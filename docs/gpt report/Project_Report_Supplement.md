# Supporting material

*The main argument is complete above. The following appendices explain the theoretical connections, reproducibility details, and checks behind that argument. They are supplementary material, not additional primary claims.*

## Appendix A. Why the endpoint and decoder controls are needed

### A.1. Probability-space minima and discrete moves

**Lemma.** A strict local minimum of $L$ over $\Delta_k^{|V|}$ is one-hot, and every recoloring of one vertex to a different color strictly increases conflicts.

**Proof.** With all rows except $v$ fixed, write $L=B+p_v^\top q_v$, where $q_v=\sum_{u\in N(v)}p_u$. If two coordinates of $p_v$ are positive, sufficiently small mass transfers between them are feasible in either direction. Different entries of $q_v$ give a decreasing direction; equal entries give a flat direction. Both contradict a strict local minimum. Thus every row is one-hot. At $p_v=e_{c_v}$, moving toward $e_a$ changes the loss by $t[M_v(a)-M_v(c_v)]$. Strictness requires this difference to be positive for every $a\ne c_v$. ∎

The implication applies to the full probability domain, not necessarily the neural image $P_\theta$. Finite logits do not produce exact simplex vertices mathematically. Non-strict minima can also contain soft rows with tied costs. These distinctions are why we test hard outputs directly.

### A.2. Confidence is not a stopping certificate

For the Potts objective,

$$
\frac{\partial L}{\partial z_{v,a}}
=p_v(a)\left[q_v(a)-p_v^\top q_v\right].
$$

Near a one-hot row, alternative probabilities are small and the bracket for the selected color approaches zero. A useful probability-space or hard recoloring direction can therefore coexist with a small logit gradient. This identity motivates a saturation interpretation, not a complete causal account of training: dropout, shared parameters, adaptive updates, and weight decay remain active.

The exploratory trajectory check is an important counterweight. From 50k to 100k, PI-GNN's median graph-level aligned partition-change fraction is 29.8%, after removing arbitrary global color permutations. High confidence does not imply fixed assignments. Moreover, some severe failures have nearly uniform probabilities and one-color argmax outputs; they are distinct from confidently improvable choices. Full diagnostics are in [RQ1](RQ1_Paper%202.md) and [RQ2, Appendix E](RQ2_Paper.md#e6-confidence-actual-assignment-changes-and-endpoint-quality).

### A.3. What D-CE guarantees

When fixing vertex $v$, D-CE chooses a minimum-cost color against already fixed neighbors and the probability distributions of unfixed neighbors. This minimum is at most the current probability-weighted average, so conditional expected conflicts do not increase. At termination the expectation equals the hard count, giving $C(\mathrm{DCE}(P))\leq L(P)$ in exact arithmetic. Implementation ties use a numerical tolerance.

For exactly uniform probabilities, unfixed neighbors contribute an equal cost to every color. With the same saturation order and tie rules, this reduces to fixed-$k$ DSATUR. The experimental near-uniform control instead uses $\operatorname{softmax}(0.1\Xi)$ for independent standard Gaussian $\Xi$, so its small perturbations can change choices. It is not numerically identical to deterministic DSATUR.

D-CE on a hard assignment is itself a sequential recoloring procedure and can take neutral choices. Consequently, its improvement is not automatically evidence of useful residual soft probability mass. The missing hardened-input control limits that attribution.

## Appendix B. Reproducibility and optimization settings

All main neural comparisons use the same graph manifest, palettes, seeds, conflict counter, and strict-descent implementation. Evaluation disables dropout. Each method is optimized per instance; weights are not trained across the DIMACS test graphs.

| Method | Optimizer | Learning rate | Embedding / hidden width | Dropout | Weight decay |
| --- | --- | ---: | --- | ---: | ---: |
| PI-GNN | AdamW | 0.0196158 | 85 / 30 | 0.465589 | 0.01 |
| Free logits | AdamW | 0.0427440 | Direct $n\times k$ logits | 0 | 0.0638583 |
| Node MLP | AdamW | 0.0280629 | 55 / 39 | 0.202060 | 0.01 |
| Fixed neural linearization | Adam | 0.00215821 | Initial PI-GNN Jacobian | 0 | 0 |

The independent search uses 43 configurations at 10k steps with two seeds, followed by ten configurations at 50k with three seeds, on 13 synthetic tuning graphs. The second stage contains the top six configurations plus four new draws. Selection minimizes mean ranks of seed-averaged final raw conflicts across tuning graph–palette cells. Tuning palettes include metadata-based feasible and one-fewer-color settings; this does not constitute a full feasible-palette DIMACS evaluation.

The linearized model is $z=z_0+J_0\delta$, using the frozen evaluation-mode Jacobian at initialization. Its selected Adam updates make it a fixed neural map, not literal gradient descent with a constant output-space preconditioning matrix. Its failure does not rule out every possible graph preconditioner.

The nine optimization arms provide 3,060 common-budget endpoints, including reused PI-GNN runs. The decoder study uses frozen outputs and contains 28,560 input–decoder records; sampling records summarize 32 trials. These are implementation coverage counts, not numbers of statistically independent graph observations.

## Appendix C. Supporting experiments and the doubts they address

| Potential alternative explanation | Check and observed outcome | What the check establishes |
| --- | --- | --- |
| Plain logits only lose after repair because they received too little time. | At PI-GNN's recorded training-time target, free logits have 17 repaired wins, 26 ties, and 25 losses versus PI-GNN; median $d_G=+5.95\%$. | The qualitative paired result persists under this timing protocol; hardware and total-cost limitations remain. |
| One fortunate repair tie seed creates PI-GNN's advantage. | Average ten new descent tie seeds per saved start. PI-GNN versus free logits changes from 26/25/17 to 26/26/16. Noise-2 versus PI-GNN remains favorable at 42/23/3. | The central comparison survives the tested tie sensitivity; all possible descent paths are not exhausted. |
| GNN initialization alone explains the direct-logit result. | Same-start logits versus row-centered, scale-matched Gaussian initialization: 14 wins, 42 ties, 12 losses; median −0.235%, $p=0.0198$. | No directional initialization advantage under the primary rule; not an equivalence result. |
| Shared neural weights are sufficient without graph aggregation. | PI-GNN beats the tuned node MLP after repair on 28 graphs and loses on five. | The tested MLP does not reproduce the advantage; widths, dropout, capacity, and initialization also differ. |
| A frozen initial neural coupling explains the advantage. | Fixed linearization has no wins and 58 losses against repaired PI-GNN, and no wins and 53 losses against plain logits. | This tuned fixed map is insufficient; no universal statement about preconditioners follows. |
| Slightly longer training resolves unfinished neural repair. | On 13 selected graphs extended to 300k, repaired PI-GNN means decrease on 12, but only one of 65 raw outputs is single-flip optimal. | Quality can improve without reaching the discrete endpoint; the selected subset is not an all-graph 300k estimate. |
| A local endpoint is already a competitive solution. | Random-start TabuCol beats repaired PI-GNN on 41 graphs, ties on 13, and loses on 14. It wins on all 12 queen graphs despite their high PI-GNN endpoint frequency. | Single-flip optimality and solution quality are distinct. |

The TabuCol reference uses ten seeds per graph and a search-time budget equal to mean PI-GNN training time, on a CPU core versus a GPU. It reports each run's best solution, while PI-GNN uses its final checkpoint and receives additional repair time. All 14 repaired PI-GNN wins occur on register-allocation graphs, where the random-start TabuCol implementation is weak and DSATUR is stronger than PI-GNN. This is a qualified quality reference, not an oracle or a controlled equal-hardware race.

Structural and family analyses remain exploratory. PI-GNN's relative repaired advantage over plain logits is associated with degree heterogeneity, but its strength depends on normalization and family composition. For example, the degree-CV correlation falls from 0.593 for relative effects to 0.126 for edge-normalized effects. These observations motivate controlled follow-ups rather than a validated method-selection rule.

Complete protocols and tables: [RQ1](RQ1_Paper%202.md), [RQ2](RQ2_Paper.md), and [RQ2 exploratory analysis](RQ2_Paper.md#exploratory-appendix-family-structure-repair-and-trajectory-analyses).

## Appendix D. Decoder robustness and computational cost

Applying common D-CE plus descent narrows the optimizer spread but preserves consequential differences. PI-GNN beats plain free logits on 30 graphs, ties on 24, and loses on 14. Noise-2 free logits beat PI-GNN on 35, tie on 29, and lose on four. Thus a better common decoder does not explain away the optimizer comparison.

For trained PI-GNN, saturation-order versus random-order D-CE gives four practical wins, 64 ties, and no losses after repair. Best-of-32 repaired sampling versus repaired argmax gives two wins, 65 ties, and one loss, with zero median difference. Sampling has more search attempts and is not a matched-cost comparison.

D-CE's modest quality gain is inexpensive in the recorded implementation. Across 340 trained-input runs, summed decoding-plus-repair time is 21.32 seconds for saturation-order D-CE, 30.39 seconds for argmax, and 789.34 seconds for the 32-sample procedure. D-CE takes more decoding work but leaves less for descent. These are recorded operation-time sums on shared CPU tasks, excluding neural training and loading; they are not isolated-machine throughput benchmarks. Full input controls, family comparisons, timings, and limitations are in [RQ3](RQ3_Paper.md).

## Appendix E. Evidence map and submission preparation

| Report component | Detailed evidence |
| --- | --- |
| Endpoint trajectories and confidence | [RQ1 detailed report](RQ1_Paper%202.md) |
| Same-start comparisons, noise, and architecture | [RQ2 detailed report](RQ2_Paper.md) |
| Repair sensitivity and structural exploration | [RQ2 exploratory appendix](RQ2_Paper.md#exploratory-appendix-family-structure-repair-and-trajectory-analyses) |
| Decoder and learned-information controls | [RQ3 detailed report](RQ3_Paper.md) |
| Figure generation and input versions | [Presentation review and reproducibility notes](files/presentation_review_20261015/README.md) |

The linked RQ reports contain repository paths to the underlying CSVs, implementations, tests, and experiment logs. Their historical project-status sections describe different completion dates; this integrated report uses the completed results from all three questions.

**Submission preparation.** Lecture 8 specifies ACL format and five pages excluding references. This Markdown file is the integrated source report; its final page count has not been verified in ACL typesetting. Appendices are separated because the lecture does not explicitly exempt them from the limit. Confirm the supplementary-material policy, add the authors' names and affiliations, and check the rendered five-page main paper before submission. The main text includes the decisive results and limitations so its argument does not depend on reading the appendices.
