"""Build a reviewable exploratory appendix from verified analysis tables.

Does not edit the vault paper: append is a separate operation after review.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_rq2_exploratory import ROOT,OUT,ARMS,FEATURES,md

def read(name):return pd.read_csv(OUT/name)
def table(df):return md(df,digits=3)
def link(name,label=None):return f'[{label or name}]({OUT/name})'
def pct(x):return f'{100*x:.1f}%'

def main():
    f=read('graph_features.csv');rs=read('repair_summary.csv');rf=read('repair_by_family.csv')
    fc=read('family_comparisons.csv');sc=read('structure_correlations.csv');rob=read('structure_robustness.csv')
    rr=read('repair_trace_runs.csv');rg=read('repair_trace_graphs.csv');sens=read('repair_sensitivity_summary.csv')
    sw=read('assignment_switch_summary.csv');sf=read('assignment_switch_by_family.csv')
    ds=read('diagnostics_family_adjusted.csv');ts=read('trajectory_summary.csv')
    sec=[]
    sec.append(r'''## Exploratory appendix: family, structure, repair, and trajectory analyses

*Added 14 October 2026 at the user's request. These analyses were performed after inspecting the main results. They supplement the completed study and do not replace its prespecified metrics or conclusions.*

### E.1 Scope, definitions, and verification

The main exploratory analysis uses all nine arms at exactly 100,000 steps: 68 graphs × five seeds × nine arms = 3,060 endpoints. We analyze both raw conflicts and conflicts after identical strict descent. Unless explicitly identified as run or vertex counts, comparisons use five-seed means within graphs and give each graph equal weight. Register-allocation graphs are pooled for display, but within-family adjustment uses the original 14 manifest families; SocialNet, Mycielski, and the single “other” graph remain separate. The pooling is descriptive, not a claim that all register-allocation graphs are independent examples of one distribution.

Four analyses were completed: (1) structural predictors of paired performance; (2) repair gains, conflict concentration, newly available recolorings, and sensitivity to descent ties; (3) confidence and actual assignment changes over training; and (4) overlap between PI-GNN and noisy-logit benefits. We additionally examine the genuinely extended 13-graph subset separately.

The endpoint aggregates reproduce the existing common-budget per-graph CSV. Every saved final logit tensor was decoded with the original float32-softmax argmax convention. A vectorized implementation of the repository's steepest strict descent reproduces all 3,060 raw counts, repaired counts, descent-move counts, and initial improvable-vertex counts; every reproduced endpoint is single-flip optimal. This decoding detail matters near uniform probabilities, where rounding can change argmax ties. No training was rerun and no original result files were overwritten.

Structural tests comprise 14 features × eight alternatives versus A × two endpoint metrics = 224 associations. Overall correlations are Spearman rank correlations. Within-family correlations subtract original-family means from globally ranked features and outcomes; their two-sided permutation p-values use 4,999 shuffles restricted within original families. Benjamini–Hochberg q-values are computed separately across the 224 overall tests and 224 family-adjusted tests. Diagnostic scans have their own correction; robustness checks and small-family correlations are descriptive. These corrections help organize a post-hoc screen, not convert it into confirmatory evidence. Graph properties are correlated, families are small and heterogeneous, and no new graph set is used for validation.

Define the repaired A advantage as $d_A=(B-A)/\max(A,1)$, with graph-level repaired counts: positive values favor A. Other alternative-versus-A effects use $d_X=(X-A)/\max(A,1)$, so positive values favor A there too. Practical ties retain the paper's rule: absolute difference below one conflict **or** relative difference at most 5%.''')
    sec.append('### E.2 Family outcomes and structural context\n\nThe tables below report alternative wins / practical ties / A wins. They include every alternative, before and after repair, rather than only the families favoring A.')
    for metric in ['raw','repaired']:
        d=fc[(fc.metric==metric)&(fc.Y=='A')].copy()
        d['verdict']=d.apply(lambda r:f'{int(r.X_better)}/{int(r.ties)}/{int(r.Y_better)}',axis=1)
        w=d.pivot(index='family_group',columns='X',values='verdict').reindex(columns=[a for a in ARMS if a!='A'])
        w.insert(0,'Graphs',f.groupby('family_group').size());w=w.reset_index().rename(columns={'family_group':'Family'})
        sec.append(f'**{metric.capitalize()} conflicts at 100k steps.**\n\n'+table(w))
    context=f.groupby('family_group').agg(Graphs=('graph','size'),median_n=('n','median'),median_k=('k','median'),
        density=('density','median'),degree_CV=('degree_cv','median'),clustering=('clustering','median'),clique_lb_over_k=('clique_lb_over_k','median')).reset_index()
    sec.append('**Structural context: medians across graphs within each display group.**\n\n'+table(context))
    sec.append('These results separate broad failure from conditional advantage: D loses repaired comparisons across most families, while A versus plain B depends on family. SocialNet/Mycielski outcomes often have only one or a few conflicts, making the absolute practical-tie threshold especially relevant. Full family effects, including relative and edge-normalised differences, are in '+link('family_comparisons.csv')+'.')

    sec.append(r'''### E.3 Which structural properties track A's repaired advantage?

We measure size $n$, palette size $k$, density, mean degree, degree coefficient of variation (standard deviation / mean), maximum-degree / mean-degree ratio, average clustering, transitivity, mean edge-neighborhood Jaccard, mean edge-neighborhood cosine similarity, a deterministic greedy clique lower bound, that lower bound divided by $k$, maximum core number divided by $k$, and mean degree divided by $k$. The clique statistic is a lower bound produced by greedy construction from at most 64 highest-degree starting vertices, **not** the maximum clique. Neighborhood similarities use open neighborhoods. Isolated vertices are retained in clustering averages and all graph counts are checked against the manifest.

The central structural screen uses $d_A$: positive correlation means increasing the feature is associated with greater relative advantage for A after repair.''')
    central=sc[(sc.metric=='repaired')&(sc.arm=='B')].set_index('feature').loc[FEATURES].reset_index()
    sec.append(table(central[['feature','rho','q_global','within_family_rho','q_within_family']]))
    sec.append(r'''The strongest overall signals are degree heterogeneity ($\rho=0.593$) and maximum-degree concentration ($\rho=0.578$), favoring A as they increase, and density ($\rho=-0.500$) and mean degree ($\rho=-0.458$), favoring B as they increase. Several relationships survive family adjustment, but the predictors are not independent mechanisms. The proposed overall neighborhood-similarity explanation has little direct support: edge Jaccard has $\rho=-0.073$ and cosine similarity $\rho=-0.014$. Within-family Jaccard is more negative, but its adjusted q-value is about 0.063. Graph size and palette size have little overall association, while palette size has a substantial negative association within families. That contrast warns against interpreting an aggregate near-zero correlation as universal irrelevance.''')
    bins=read('structural_bins.csv');b=bins[(bins.feature=='degree_cv')&(bins.metric=='repaired')&(bins.X=='B')&(bins.Y=='A')]
    b=b[['feature_min','feature_max','graphs','Y_better','ties','X_better','median_d']].rename(columns={'Y_better':'A wins','X_better':'B wins','median_d':'Median A advantage'})
    sec.append('**Degree-heterogeneity tertiles, using feature quantiles rather than outcome-selected thresholds.**\n\n'+table(b))
    sec.append('The most heterogeneous third contains 12 A wins, 11 ties, and no B wins. However, degree heterogeneity is not a validated selection rule: excluding register-allocation graphs reduces its correlation from 0.593 to 0.290; adjustment for ranked size, palette size, and density reduces it to 0.228. Across all leave-one-display-family-out checks, its correlation ranges from 0.290 to 0.706. Excluding graphs where both repaired means are below five leaves 44 graphs and a correlation of 0.441.')
    robustrows=[]
    for c in ['degree_cv','density','mean_degree','edge_jaccard','neighbour_cosine']:
        dd=rob[rob.feature==c].set_index('check')
        robustrows.append({'Feature':c,'Relative effect rho':dd.loc['relative_to_A','rho'],
            'Edge-normalised effect rho':dd.loc['edge_normalised_difference','rho'],
            'Excluding both endpoints <5':dd.loc['exclude_both_endpoints_below_5','rho'],
            'Partial rho, size/palette/density':dd.loc['partial_rank_control_n_k_density_except_self','rho']})
    sec.append('**Sensitivity to scale and covariates.** The feature itself is omitted from its control set.\n\n'+table(pd.DataFrame(robustrows)))
    sec.append(r'Degree heterogeneity has only $\rho=0.126$ with the difference in conflicts per edge, versus 0.593 with relative conflicts. Density also changes strength across summaries. Thus, the observed regime concerns relative endpoint quality, especially where conflict totals are small; it is not evidence of large edge-normalised gains or a unique structural cause. Symmetric relative normalisation gives the same Spearman ordering here, but edge normalisation asks a different question.')
    sec.append('**Complete overall structural screen.** Each entry is Spearman rho for alternative versus A; positive values mean that increasing the feature favors A. Adjusted q-values and family-centered versions for every entry are retained in '+link('structure_correlations.csv')+'.')
    for metric in ['raw','repaired']:
        matrix=sc[sc.metric==metric].pivot(index='feature',columns='arm',values='rho').reindex(index=FEATURES,columns=[a for a in ARMS if a!='A']).reset_index()
        sec.append(f'**{metric.capitalize()} outcome correlations.**\n\n'+table(matrix))
    sec.append('Feature-by-family correlations are available in '+link('structure_within_groups.csv')+' for display groups containing at least four graphs; they are descriptive and should not be ranked by small-group p-values. All continuous-feature bins and robustness checks are in '+link('structural_bins.csv')+' and '+link('structure_robustness.csv')+'.')

    sec.append('### E.4 How much repair helps, and how it helps\n\nThe following counts describe strict improvement within a run; the between-method practical-tie threshold is not applied. All free-logit variants are unchanged by repair at this budget.')
    show=rs[['arm','improved_runs','unchanged_runs','raw_mean','repaired_mean','mean_removed','median_removed_if_helped','graphs_any_help']].copy()
    sec.append(table(show))
    sec.append('A and A-no-dropout lose 92.4% and 92.6% of their aggregate raw conflicts through repair, compared with 9.6% for C and 24.4% for D. These percentages are weighted by raw conflict totals and can be dominated by severe failures; they are not typical per-run reductions. Among improved runs, median removals are 9 for A, 23 for A-no-dropout, 4 for C, and 13 for D.')
    help=rf[rf.arm.isin(['A','A_nodrop','C','D'])].pivot(index='family_group',columns='arm',values='helped_fraction').reindex(columns=['A','A_nodrop','C','D']).map(pct).reset_index()
    sec.append('**Fraction of runs helped by repair, by family.**\n\n'+table(help))
    removed=rf[rf.arm.isin(['A','A_nodrop','C','D'])].pivot(index='family_group',columns='arm',values='median_fraction_removed').reindex(columns=['A','A_nodrop','C','D']).map(pct).reset_index()
    sec.append('**Median across graphs of seed-mean fraction of raw conflicts removed.**\n\n'+table(removed))
    sec.append('Queen and register-allocation graphs illustrate different routes to A’s advantage. Only 15.0% of queen A runs improve under descent, whereas 97.1% of register-allocation A runs improve. In both families A often beats plain B after repair. A single explanation based on “needing more cleanup” therefore does not cover its advantage.')
    tr=[]
    for arm in ['A','A_nodrop','C','D']:
        d=rr[(rr.arm==arm)&(rr.moves>0)]
        tr.append({'Arm':arm,'Improved runs':len(d),'Median vertices recoloured':pct(d.unique_recoloured_fraction.median()),
            'Runs recolouring initially non-improvable vertices':int((d.newly_improvable_recoloured>0).sum()),
            'Runs recolouring a vertex more than once':int((d.repeated_moves>0).sum()),
            'Median raw top-10% conflict share':pct(d.conflict_top10_share.median())})
    sec.append('**Trace of repair on improved runs.** The top 10% statistic is the fraction of conflicting-edge endpoint incidences carried by the most conflicted 10% of vertices. “Initially non-improvable” means a vertex whose improving recoloring became available after earlier moves.\n\n'+table(pd.DataFrame(tr)))
    sec.append('Most improved A and C runs recolour only a small share of vertices (medians 2.2% and 1.6%). Nonetheless, repair is not always a one-pass cleanup: initially non-improvable vertices are recoloured in 56 A runs, 78 C runs, and 130 D runs. D has a particularly substantial cascading component. These traces describe how descent unfolds; they do not establish that cascades cause better final minima. Conflict concentration alone is also insufficient: already-optimal B outputs can concentrate their few residual conflicts strongly while admitting no improving move. Full run, graph, family, and correlation tables are in '+link('repair_trace_runs.csv')+', '+link('repair_trace_graphs.csv')+', '+link('repair_trace_by_family.csv')+', and '+link('repair_trace_correlations.csv')+'.')

    # Separate global collapse from confident but improvable errors.
    collapse=[]
    for arm in ARMS:
        d=rr[rr.arm==arm];c=d[d.collapsed_one_colour]
        collapse.append({'Arm':arm,'One-colour runs':len(c),'Share of total raw conflicts':pct(c.raw.sum()/d.raw.sum())})
    sec.append('**A distinct raw-output failure mode: one-colour collapse.**\n\n'+table(pd.DataFrame(collapse)))
    sec.append('A’s four collapsed outputs are all DSJR500.5, seeds 0, 1, 2, and 4; they alone account for 45.6% of A’s total raw conflicts. A-no-dropout has 15 collapsed outputs accounting for 83.8% of its raw conflicts. No such output occurs for C, D, or any free-logit arm. This helps explain why raw aggregate means are so different despite less dramatic graph-wise comparisons. It also shows that neural failures are heterogeneous rather than uniformly saturated.')
    mv=pd.read_csv(ROOT/'results/analysis/exp3_movement_tidy.csv');mv=mv[mv.epoch==100000]
    collapsed=rr[rr.collapsed_one_colour].merge(mv[['arm','graph','seed','margin_median','frac_pmax_099','eval_entropy']],on=['arm','graph','seed'])
    collapsed['normalised_entropy']=collapsed.eval_entropy/np.log(collapsed.k)
    collapsed.to_csv(OUT/'collapsed_outputs.csv',index=False)
    sec.append(table(collapsed[['arm','graph','seed','k','raw','repaired','frac_pmax_099','normalised_entropy']]))
    sec.append(r'Eighteen of these 19 collapsed outputs have entropy at least 99% of the uniform entropy and no vertices with $p_{\max}\geq0.99$; one A-no-dropout queen7_7 output is confidently collapsed. Thus, almost-uniform probabilities and saturated incorrect assignments are distinct endpoint phenomena.')

    sec.append('### E.5 Is the repaired ranking sensitive to random tie-breaking?\n\nFor each saved raw coloring we evaluate ten new descent tie seeds, 10000–10009, and average their conflict counts rather than selecting the best. Outputs already single-flip optimal are reused unchanged. This yields 30,600 tie-seed outcomes, with 8,770 new descents on the 877 improvable outputs. The original run-seed descent remains the reference.')
    sec.append(table(sens[['arm','variable_tie_runs','mean_tie_range','mean_original_conflicts','mean_tie_conflicts']]))
    comp=read('repair_sensitivity_comparisons.csv').copy()
    comp['Outcome']=comp.metric.map({'repaired':'Original descent','tie_mean':'Mean of ten descents'})
    sec.append('**Paired comparisons after averaging tie-breaks.** Counts favor X versus Y.\n\n'+table(comp[['X','Y','Outcome','X_better','ties','Y_better','median_d']]))
    sec.append('Individual endpoints can depend on tie-breaking: conflict counts vary in 104 A runs, 199 A-no-dropout runs, 75 C runs, and 130 D runs. Nevertheless, the central ordering is stable. Plain B versus A changes from 17/25/26 to 16/26/26; C versus A changes from 5/35/28 to 4/36/28; D still has no wins; noise-2 remains strongly favorable at 42/23/3. This makes a lucky single descent tie seed an inadequate explanation of the main results. Ten samples do not exhaust all possible descent paths. Per-graph changed verdicts and family-specific sensitivity are retained in '+link('repair_sensitivity_verdicts.csv')+' and '+link('repair_sensitivity_changed_graphs.csv')+'.')

    sec.append('### E.6 Confidence, actual assignment changes, and endpoint quality\n\nHigh confidence is not equivalent to a fixed coloring. We measure both confidence from saved probability diagnostics and actual changes between saved colorings. For label-invariant comparison, we optimally align color labels between checkpoints using a maximum-weight assignment; the reported partition-change fraction counts vertices still different after that alignment. Edge-status changes count edges switching between conflicting and nonconflicting. These are endpoint-to-endpoint changes and cannot recover intervening switches.')
    sec.append('**First saved checkpoint with at least 90% of vertices having $p_{\\max}\\geq0.99$.** Step medians are across runs that reach the threshold; never-reaching runs are counted separately. Checkpoint spacing makes these observed crossing times, not exact event times.\n\n'+table(ts[['arm','runs','ever_confident','median_first_step','stable_confident','median_stable_step','reversal_runs']]))
    late=sw[sw.from_step==50000][['arm','vertex_switch_fraction','partition_switch_fraction','edge_conflict_status_switch']].copy()
    for c in late.columns[1:]:late[c]=late[c].map(pct)
    sec.append('**Changes from 50k to 100k steps: median across graph-level seed means.**\n\n'+table(late))
    sec.append('Although A and C become confident early, their late color partitions still change substantially: the median aligned changed-vertex fractions are 29.8% for A and 25.1% for C. Plain B and B-rand have median zero late changes, while noise-2 has 27.2%. Therefore, “early confidence means the neural coloring freezes” is not supported. High confidence can coexist with continued exploration or switches; late movement by itself does not guarantee better endpoints. Relatively few edges change conflict status compared with the number of changed vertices, so assignment movement and conflict-count progress should be distinguished.')
    latefam=sf[sf.from_step==50000].pivot(index='family_group',columns='arm',values='partition_switch_fraction').reindex(columns=['A','A_nodrop','B','B_noise2','C','D']).map(pct).reset_index()
    sec.append('**Late aligned partition changes by family.**\n\n'+table(latefam))
    early=ds[(ds.diagnostic=='confidence100')&(ds.target=='repaired_per_edge')]
    sec.append('**Confidence at step 100 versus final repaired conflicts per edge.** Positive rho means more early confidence is associated with worse final edge-normalised outcomes.\n\n'+table(early[['arm','rho','within_family_rho','partial_rho_n_k_density']]))
    sec.append('Every arm shows a positive overall and family-centered association, but controlling ranked size, palette size, and density substantially weakens or reverses it for several arms: A falls from 0.715 to 0.247, B changes from 0.629 to −0.160, and noise-2 from 0.498 to −0.378. Early confidence is consequently a descriptive regime marker, not evidence that commitment causes failure. The structural adjustment is itself an exploratory linear adjustment on ranks, not a causal model.')
    confidence=[]
    for arm in ['A','A_nodrop','C','D']:
        d=rr[rr.arm==arm]
        confidence.append({'Arm':arm,'Improvable run-vertices':int(d.improvable_vertices.sum()),
            'Also confident':int(d.confident_improvable_vertices.sum()),
            'Confident share':pct(d.confident_improvable_vertices.sum()/d.improvable_vertices.sum()),
            'Runs with confident improvable vertices':int((d.confident_improvable_vertices>0).sum())})
    sec.append('**Confidence at vertices that admit a strict improving recoloring at 100k.** These are pooled run-vertex counts, not independent statistical samples.\n\n'+table(pd.DataFrame(confidence)))
    sec.append('For C, 99.7% of initially improvable run-vertices are already above the 0.99 confidence threshold. This directly demonstrates saturated but locally improvable outputs without message passing. A has both confident residual errors and almost-uniform collapses, producing a lower pooled confident share. The diagnostic is a mismatch between softmax confidence and discrete improvement, not a proof of convergence failure in parameter space. All diagnostic associations, including family adjustment and normalised palette usage, are in '+link('diagnostics_family_adjusted.csv')+'; trajectory and assignment summaries by family are in '+link('trajectory_by_family.csv')+' and '+link('assignment_switch_by_family.csv')+'.')

    sec.append(r'''### E.7 Do PI-GNN and logit noise help on the same graphs?

Define benefits relative to the same plain-B baseline:

$$b_A=\frac{B-A}{\max(B,1)},\qquad b_{\mathrm{noise}}=\frac{B-B_{\mathrm{noise}}}{\max(B,1)}.$$

Positive values mean fewer repaired conflicts than B. This avoids comparing two effects both divided by A, but the benefits still share a baseline and are not independent observations. We also examine edge-normalised benefits.''')
    sec.append(table(read('noise_overlap_robustness.csv')))
    sec.append(r'Noise-1 has moderate overall overlap with A’s relative benefit ($\rho=0.334$), but it vanishes after within-family adjustment ($\rho=-0.139$). Noise-2’s overall relative-benefit correlation is only 0.110; the within-family value is −0.177. Edge normalisation yields different within-family associations (0.424 for noise-1 and 0.356 for noise-2). The overlap therefore depends on scale and grouping and does not identify a common mechanism.')
    overlap=read('noise_overlap_verdicts.csv')
    overlap['B_vs_A']=overlap.B_vs_A.map({'X better':'B better','Y better':'A better','tie':'Practical tie'})
    sec.append('**Noise-2 versus B, stratified by the repaired B-versus-A verdict.** “B better” means plain B beats A; “A better” means A beats B.\n\n'+table(overlap.rename(columns={'B_vs_A':'B versus A','X better':'Noise-2 beats B','tie':'Noise-2 ties B'})))
    sec.append('Noise-2 beats B on 25 of the 26 graphs where A beats B and on 16 of the 17 graphs where B beats A, with no practical losses to B. It therefore helps beyond the subset where PI-GNN has an advantage. Family-specific noise-benefit correlations are in '+link('noise_overlap_correlations.csv')+'; with only 2–14 graphs per group, perfect or large small-group correlations should not be treated as mechanistic evidence.')

    sec.append('### E.8 Longer training on the same selected graphs\n\nThe earlier mixed final-budget result preserves the overall ranking, but this does not mean individual methods stop improving. Here we isolate the 13 graphs with A extended from 100k to 300k and compare each graph with itself at the two budgets. Only graph-arm pairs with all five seeds at both checkpoints enter the table. Counts below compare 300k against 100k within the same method; they are not A-versus-B wins.')
    sec.append(table(read('extended_subset_summary.csv')))
    sec.append('A improves its repaired endpoint under the practical rule on six of the 13 selected graphs, ties on six, and worsens on one. Plain B, B-rand, and noise-0.5 are unchanged on all 13; noise-1 improves on seven, and noise-2 on five, with no practical worsening for those noisy arms. Thus, the unchanged aggregate ordering coexists with continuing progress for A and stronger-noise logits. C and A-no-dropout have no 300k coverage. D has complete paired coverage for only two extended graphs and must not be compared as if it covered the same 13. Selection of the extension set limits generalisation. Exact graph-level values and coverage are in '+link('extended_subset_graphs.csv')+'.')

    sec.append('### E.9 What these analyses add to RQ2\n\nThe clearest additions are that A’s repaired advantage is associated with structural regime and effect scale, repair tie-breaking does not explain away the main ranking, early confidence does not imply frozen assignments, and confident local errors occur in the MLP as well as PI-GNN. Noise-2 improves plain B across both sides of the A-versus-B split. The data also expose two distinct neural endpoint failures: nearly uniform one-colour collapse and confidently improvable colorings.\n\nThe structural evidence motivates a narrower hypothesis—relative neural benefit may be associated with degree heterogeneity and density—but does not establish those properties as causes or a reliable method-selection policy. The weak overall neighborhood-similarity findings do not support elevating that proposed explanation. A small, independently designed graph set varying degree heterogeneity and density while controlling size and palette would be needed to test the structural hypothesis. Current post-hoc findings remain candidates for later inclusion, not new primary conclusions.')
    sec.append('![Exploratory structure, benefit overlap, and confidence trajectories](files/rq2_exploratory_overview.png)\n\n*Exploratory overview. Left: degree heterogeneity versus repaired A advantage. Middle: A and noise-2 benefits with a common B baseline. Right: median graph-level seed-mean confidence trajectories. These plots illustrate association and trajectory differences, not causal effects.*')
    sec.append('### E.10 Reproducible artifacts and audit\n\nThe full outputs are retained in '+f'[the exploratory analysis directory]({OUT})'+'. The three computation scripts and report generator are saved alongside the repository scripts:')
    scripts=['analyze_rq2_exploratory.py','analyze_rq2_repair_sensitivity.py','analyze_rq2_robustness.py','report_rq2_exploratory.py']
    sec.append('\n'.join(f'- [{name}]({ROOT/"scripts"/name})' for name in scripts))
    sec.append('Inputs are the graph manifest and `.col` files, the existing Experiment 3 tidy/per-graph/movement CSVs, and the saved checkpoint `Z` tensors. Run the computation scripts in the order above, then the report generator. Required packages are NumPy, pandas, SciPy, NetworkX, PyTorch, and matplotlib; the scripts are CPU-only. No training, best-of selection, or change to the original primary-rule CSVs is performed. SHA-256 hashes of aggregate inputs and script files are saved in '+link('input_hashes.json')+'.\n\nCompleted checks: 68 graphs match manifest sizes/edge counts; nine arms each have 340 unique common-budget endpoints; all original endpoint quantities reproduce; all 30,600 tie outcomes are non-worsening and single-flip optimal; all extended-subset means use five seeds at both budgets. Missing/constant correlations are shown as dashes rather than zero. Selected exploratory q-values have a permutation resolution floor of 0.0002; no stronger numerical precision or population-wide interpretation is implied.')
    report='\n\n'.join(sec)+'\n'
    (OUT/'appendix.md').write_text(report)
    hashes={}
    for p in [ROOT/'configs/sweep_manifest.csv',ROOT/'results/analysis/exp3_tidy.csv',ROOT/'results/analysis/exp3_per_graph.csv',ROOT/'results/analysis/exp3_movement_tidy.csv']+[ROOT/'scripts'/s for s in scripts]:
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'input_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print('Report saved:',OUT/'appendix.md','words',len(report.split()))

if __name__=='__main__':main()
