"""Reproducible post-hoc RQ2 diagnostics; writes only a separate exploratory folder.

Run with the project's scientific Python dependencies. No training or original
result files are changed. The report is generated separately before vault append.
"""
from pathlib import Path
import sys, json, warnings, hashlib
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis/rq2_exploratory_20261014'
OUT.mkdir(exist_ok=True)
ARMS = ['A', 'A_nodrop', 'B', 'Brand', 'B_noise0.5', 'B_noise1', 'B_noise2', 'C', 'D']
FEATURES = ['n', 'density', 'k', 'mean_degree', 'degree_cv', 'max_degree_over_mean',
            'clustering', 'transitivity', 'edge_jaccard', 'neighbour_cosine',
            'greedy_clique_lb', 'clique_lb_over_k', 'max_core_over_k', 'mean_degree_over_k']

def group_family(f):
    return 'register' if f in ['fpsol', 'inithx', 'mulsol', 'zeroin'] else f

def bh(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ids = np.flatnonzero(np.isfinite(p)); order = ids[np.argsort(p[ids])]
    if len(order): out[order] = np.minimum(1, np.minimum.accumulate((p[order] * len(order) / np.arange(1, len(order)+1))[::-1])[::-1])
    return out

def corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4 or np.ptp(np.asarray(x)[mask]) == 0 or np.ptp(np.asarray(y)[mask]) == 0: return np.nan, np.nan
    r = spearmanr(np.asarray(x)[mask], np.asarray(y)[mask])
    return float(r.statistic), float(r.pvalue)

def md(df, cols=None, digits=3):
    if cols: df = df[cols]
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            if not np.isfinite(x): return '—'
            return f'{x:.{digits}f}' if abs(x) >= .001 or x == 0 else f'{x:.2e}'
        return str(x)
    return '\n'.join(['| ' + ' | '.join(map(str, df.columns)) + ' |', '| ' + ' | '.join(['---']*len(df.columns)) + ' |'] + ['| ' + ' | '.join(fmt(x) for x in row) + ' |' for row in df.itertuples(index=False, name=None)])

def graph_load(name):
    n = None; edges = set()
    for line in (ROOT / 'data/dimacs/col' / f'{name}.col').open():
        t = line.split()
        if not t: continue
        if t[0] == 'p': n = int(t[2])
        elif t[0] == 'e':
            u,v = int(t[1])-1,int(t[2])-1
            if u != v: edges.add(tuple(sorted((u,v))))
    g = nx.Graph(); g.add_nodes_from(range(n)); g.add_edges_from(sorted(edges))
    return g

def structural(man):
    rows=[]
    for _,r in man.iterrows():
        g=graph_load(r.graph); n=g.number_of_nodes(); m=g.number_of_edges(); k=int(r.k)
        assert (n,m)==(r.n,r.m),r.graph
        a=nx.to_numpy_array(g,dtype=np.float64); deg=a.sum(1); common=a@a
        u,v=np.array(list(g.edges())).T
        tri=(a*common).sum(1)/2
        local=np.divide(2*tri,deg*(deg-1),out=np.zeros(n),where=deg>1)
        wedge=(deg*(deg-1)/2).sum()
        union=deg[u]+deg[v]-common[u,v]
        # Open-neighbourhood Jaccard: adjacent endpoints are included in the union.
        jac=np.divide(common[u,v],union,out=np.zeros(m),where=union>0)
        cosine=common[u,v]/np.sqrt(deg[u]*deg[v])
        clique=1
        # Deterministic bounded greedy lower bound, not maximum-clique estimation.
        starts=sorted(g,key=lambda x:(-deg[x],x))[:64]
        for start in starts:
            size=1; candidates=set(g[start])
            while candidates:
                node=max(candidates,key=lambda x:(len(candidates.intersection(g[x])),deg[x],-x))
                size+=1; candidates.intersection_update(g[node])
            clique=max(clique,size)
        core=max(nx.core_number(g).values())
        rows.append(dict(graph=r.graph,family=r.family,family_group=group_family(r.family),n=n,m=m,k=k,
            density=2*m/(n*(n-1)),mean_degree=deg.mean(),degree_cv=deg.std()/deg.mean(),
            max_degree_over_mean=deg.max()/deg.mean(),clustering=local.mean(),
            transitivity=tri.sum()/wedge if wedge else 0,edge_jaccard=jac.mean(),neighbour_cosine=cosine.mean(),
            greedy_clique_lb=clique,clique_lb_over_k=clique/k,max_core_over_k=core/k,mean_degree_over_k=deg.mean()/k,
            isolates=int((deg==0).sum())))
        print('structure',r.graph,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'graph_features.csv',index=False);return df

def comparison_table(pg):
    rows=[]; wide=pg.pivot(index='graph',columns='arm',values=['conflicts','conflicts_after_descent'])
    for metric, label in [('conflicts','raw'),('conflicts_after_descent','repaired')]:
        for x,y in [(a,'A') for a in ARMS if a!='A']+[('B_noise1','B'),('B_noise2','B'),('C','B'),('D','B')]:
            xx=wide[metric][x]; yy=wide[metric][y]
            d=(xx-yy)/np.maximum(yy,1);tie=(abs(xx-yy)<1)|(abs(d)<=.05)
            for graph in wide.index:
                rows.append(dict(graph=graph,metric=label,X=x,Y=y,d=d[graph],
                    verdict='tie' if tie[graph] else ('X better' if xx[graph]<yy[graph] else 'Y better'),
                    X_value=xx[graph],Y_value=yy[graph],edge_delta=(xx[graph]-yy[graph])/pg.set_index('graph').loc[graph,'m'].iloc[0]))
    return pd.DataFrame(rows)

def structure_correlations(features, comp):
    # All 8 alternatives vs A, both endpoints: 224 feature-outcome associations.
    effects=comp[comp.Y=='A'].pivot(index='graph',columns=['metric','X'],values='d').sort_index()
    f=features.set_index('graph').loc[effects.index];xr=np.array([rankdata(f[c]) for c in FEATURES]).T
    yr=np.array([rankdata(effects[c]) for c in effects.columns]).T
    xr-=xr.mean(0);yr-=yr.mean(0)
    xc=xr.copy();yc=yr.copy(); groups=[np.flatnonzero(f.family.to_numpy()==fam) for fam in sorted(f.family.unique())]
    for ids in groups: xc[ids]-=xc[ids].mean(0);yc[ids]-=yc[ids].mean(0)
    den=np.sqrt((xc*xc).sum(0)[:,None]*(yc*yc).sum(0)[None,:])
    with np.errstate(invalid='ignore',divide='ignore'): observed=xc.T@yc/den
    counts=np.zeros_like(observed);rng=np.random.default_rng(20261014)
    for _ in range(4999):
        order=np.arange(len(f))
        for ids in groups: order[ids]=rng.permutation(ids)
        with np.errstate(invalid='ignore',divide='ignore'): perm=xc.T@yc[order]/den
        counts+=abs(perm)>=abs(observed)-1e-12
    rows=[]
    for i,c in enumerate(FEATURES):
        for j,(metric,arm) in enumerate(effects.columns):
            rho,p=corr(f[c].to_numpy(),effects[(metric,arm)].to_numpy())
            rows.append(dict(feature=c,metric=metric,arm=arm,rho=rho,p=p,within_family_rho=observed[i,j],
                within_family_p=(counts[i,j]+1)/5000 if np.isfinite(observed[i,j]) else np.nan))
    out=pd.DataFrame(rows);out['q_global']=bh(out.p);out['q_within_family']=bh(out.within_family_p)
    # Family-specific descriptive correlations only, no small-group hypothesis tests.
    fam=[]
    for fg,ids in f.groupby('family_group').groups.items():
        if len(ids)<4: continue
        for c in FEATURES:
            for metric,arm in effects.columns:
                rho,_=corr(f.loc[ids,c].to_numpy(),effects.loc[ids,(metric,arm)].to_numpy())
                fam.append(dict(family_group=fg,graphs=len(ids),feature=c,metric=metric,arm=arm,rho=rho))
    pd.DataFrame(fam).to_csv(OUT/'structure_within_groups.csv',index=False)
    out.to_csv(OUT/'structure_correlations.csv',index=False)
    return out

def main():
    man=pd.read_csv(ROOT/'configs/sweep_manifest.csv').rename(columns={'name':'graph'})
    feat=structural(man)
    tidy=pd.read_csv(ROOT/'results/analysis/exp3_tidy.csv')
    ep=tidy[tidy.epoch==100000].copy()
    assert len(ep)==3060 and not ep.duplicated(['arm','graph','seed']).any()
    assert ep.groupby('arm').size().eq(340).all()
    ep=ep.merge(feat,on=['graph','k'],validate='many_to_one')
    ep['removed']=ep.conflicts-ep.conflicts_after_descent
    assert ep.removed.ge(0).all()
    ep['helped']=ep.removed>0
    ep['repair_fraction']=ep.removed/ep.conflicts.clip(lower=1)
    ep['moves_per_vertex']=ep.descent_moves/ep.n
    ep['initial_improvable_fraction']=ep.n_improvable/ep.n
    ep['removed_per_move']=ep.removed/ep.descent_moves.replace(0,np.nan)
    pg=ep.groupby(['arm','graph','family','family_group'],as_index=False).mean(numeric_only=True)
    saved=pd.read_csv(ROOT/'results/analysis/exp3_per_graph.csv')
    saved=saved[(saved.budget=='100000')&(saved.view=='steps')]
    check=pg.merge(saved,on=['arm','graph'],suffixes=('_new','_saved'),validate='one_to_one')
    for c in ['conflicts','conflicts_after_descent']: assert np.allclose(check[c+'_new'],check[c+'_saved'],atol=1e-9)
    ep.to_csv(OUT/'endpoint_run_metrics.csv',index=False);pg.to_csv(OUT/'endpoint_graph_metrics.csv',index=False)
    comp=comparison_table(pg).merge(feat[['graph','family','family_group']],on='graph',validate='many_to_one')
    comp.to_csv(OUT/'paired_effects.csv',index=False)
    famcomp=comp.groupby(['family_group','metric','X','Y']).agg(graphs=('graph','size'),median_d=('d','median'),
        X_better=('verdict',lambda x:(x=='X better').sum()),ties=('verdict',lambda x:(x=='tie').sum()),
        Y_better=('verdict',lambda x:(x=='Y better').sum()),mean_edge_delta=('edge_delta','mean')).reset_index()
    famcomp.to_csv(OUT/'family_comparisons.csv',index=False)
    repair=[]
    for arm in ARMS:
        rr=ep[ep.arm==arm];gg=pg[pg.arm==arm];imp=rr[rr.helped]
        repair.append(dict(arm=arm,improved_runs=len(imp),unchanged_runs=len(rr)-len(imp),
            raw_mean=rr.conflicts.mean(),repaired_mean=rr.conflicts_after_descent.mean(),
            mean_removed=rr.removed.mean(),median_removed_if_helped=imp.removed.median(),
            mean_moves=rr.descent_moves.mean(),graphs_any_help=int((gg.helped>0).sum()),
            graphs_all_help=int((gg.helped==1).sum()),aggregate_reduction=rr.removed.sum()/rr.conflicts.sum()))
    repair=pd.DataFrame(repair);repair.to_csv(OUT/'repair_summary.csv',index=False)
    rf=pg.groupby(['family_group','arm']).agg(graphs=('graph','size'),helped_fraction=('helped','mean'),
        median_fraction_removed=('repair_fraction','median'),mean_removed=('removed','mean'),
        mean_moves=('descent_moves','mean'),median_improvable_fraction=('initial_improvable_fraction','median'),
        mean_repaired=('conflicts_after_descent','mean')).reset_index()
    rf.to_csv(OUT/'repair_by_family.csv',index=False)
    assoc=structure_correlations(feat,comp)

    # Benefits use a COMMON B denominator to avoid spurious shared-A-baseline effects.
    w=pg.pivot(index='graph',columns='arm',values='conflicts_after_descent')
    benefits=pd.DataFrame(index=w.index)
    benefits['A_benefit_over_B']=(w.B-w.A)/np.maximum(w.B,1)
    benefits['noise1_benefit_over_B']=(w.B-w.B_noise1)/np.maximum(w.B,1)
    benefits['noise2_benefit_over_B']=(w.B-w.B_noise2)/np.maximum(w.B,1)
    benefits=benefits.join(feat.set_index('graph')[['family','family_group']])
    benefits.to_csv(OUT/'noise_benefit_overlap.csv')
    overlap=[]
    for label,df in [('All',benefits)]+list(benefits.groupby('family_group')):
        for noise in ['noise1','noise2']:
            rho,p=corr(df.A_benefit_over_B.to_numpy(),df[noise+'_benefit_over_B'].to_numpy())
            overlap.append(dict(group=label,graphs=len(df),noise=noise,rho=rho,p=p))
    overlap=pd.DataFrame(overlap);overlap.to_csv(OUT/'noise_overlap_correlations.csv',index=False)
    av=comp[(comp.metric=='repaired')&(comp.X=='B')&(comp.Y=='A')].set_index('graph').verdict
    nv=comp[(comp.metric=='repaired')&(comp.X=='B_noise2')&(comp.Y=='B')].set_index('graph').verdict
    cross=pd.crosstab(av.rename('B_vs_A'),nv.rename('noise2_vs_B'));cross.to_csv(OUT/'noise_overlap_verdicts.csv')

    # Saved trajectories: earliest observed high confidence, endpoint repair, late changes in score.
    mv=pd.read_csv(ROOT/'results/analysis/exp3_movement_tidy.csv')
    mv=mv[mv.epoch<=100000].copy()
    assert not mv.duplicated(['arm','graph','seed','epoch']).any()
    mv['entropy_normalised']=mv.eval_entropy/np.log(mv.graph.map(feat.set_index('graph').k))
    diag=mv[mv.epoch==100000].groupby(['arm','graph'],as_index=False).mean(numeric_only=True)
    dg=diag.merge(pg[['arm','graph','family','family_group','repair_fraction','removed','helped','conflicts_after_descent']],on=['arm','graph'])
    diag_cols=['margin_median','frac_pmax_099','entropy_normalised','logit_dist_centred_rel','n_colours_used','dirichlet_per_2m']
    dfam=dg.groupby(['family_group','arm'])[diag_cols].median().reset_index();dfam.to_csv(OUT/'diagnostics_by_family.csv',index=False)
    tr=[]
    for (arm,g,s),rr in mv.groupby(['arm','graph','seed']):
        rr=rr.sort_values('epoch');hits=rr[rr.frac_pmax_099>=.9]
        endpoint=ep[(ep.arm==arm)&(ep.graph==g)&(ep.seed==s)].iloc[0]
        last=rr.iloc[-1]
        # Last observed checkpoint after which confidence remains above threshold.
        bad=rr[rr.frac_pmax_099<.9]
        stable_hits=rr[rr.epoch>(bad.epoch.max() if len(bad) else 0)]
        stable=stable_hits.epoch.min() if len(stable_hits) else np.nan
        before=rr[rr.epoch>=10000]
        tr.append(dict(arm=arm,graph=g,seed=s,first_confident_step=hits.epoch.min() if len(hits) else np.nan,
            stable_confident_step=stable,confidence_reversal=bool(len(hits) and (rr[rr.epoch>hits.epoch.min()].frac_pmax_099<.9).any()),
            confidence100=float(rr.loc[rr.epoch==100,'frac_pmax_099'].iloc[0]),
            repaired=endpoint.conflicts_after_descent,repair_fraction=endpoint.repair_fraction,
            late_raw_count_range=int(before.eval_conflicts.max()-before.eval_conflicts.min()),
            normalised_late_raw_range=(before.eval_conflicts.max()-before.eval_conflicts.min())/endpoint.m))
    tr=pd.DataFrame(tr).merge(feat[['graph','family','family_group']],on='graph')
    tr.to_csv(OUT/'trajectory_run_metrics.csv',index=False)
    tg=tr.groupby(['arm','graph','family','family_group']).mean(numeric_only=True).reset_index()
    tsummary=tr.groupby('arm').agg(runs=('graph','size'),ever_confident=('first_confident_step','count'),
        median_first_step=('first_confident_step','median'),stable_confident=('stable_confident_step','count'),
        median_stable_step=('stable_confident_step','median'),reversal_runs=('confidence_reversal','sum')).reset_index()
    tsummary.to_csv(OUT/'trajectory_summary.csv',index=False)
    tg.groupby(['family_group','arm']).median(numeric_only=True).reset_index().to_csv(OUT/'trajectory_by_family.csv',index=False)
    diag_assoc=[]
    for arm in ARMS:
        df=dg[dg.arm==arm]
        for c in diag_cols:
            for target in ['repair_fraction','conflicts_after_descent']:
                y=df[target].to_numpy()
                if target=='conflicts_after_descent': y=y/df.m.to_numpy()
                rho,p=corr(df[c].to_numpy(),y)
                diag_assoc.append(dict(arm=arm,diagnostic=c,target=target,rho=rho,p=p))
        df=tg[tg.arm==arm]
        for c in ['confidence100','first_confident_step','normalised_late_raw_range']:
            rho,p=corr(df[c].to_numpy(),(df.repaired/df.graph.map(feat.set_index('graph').m)).to_numpy())
            diag_assoc.append(dict(arm=arm,diagnostic=c,target='repaired_per_edge',rho=rho,p=p))
    da=pd.DataFrame(diag_assoc);da['q']=bh(da.p);da.to_csv(OUT/'diagnostic_correlations.csv',index=False)
    # Save plotting input and plots. Family names encode benchmark groups, not inferred mechanisms.
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    colors={g:plt.get_cmap('tab20')(i) for i,g in enumerate(sorted(feat.family_group.unique()))}
    e=comp[(comp.metric=='repaired')&(comp.X=='B')&(comp.Y=='A')].set_index('graph')
    ff=feat.set_index('graph').join(e[['d']])
    for fg,rr in ff.groupby('family_group'):
        axes[0].scatter(rr.degree_cv,rr.d,c=[colors[fg]],label=fg,s=28)
        bb=benefits[benefits.family_group==fg]
        axes[1].scatter(bb.A_benefit_over_B,bb.noise2_benefit_over_B,c=[colors[fg]],s=28)
    axes[0].axhline(0,color='grey',lw=1);axes[0].set(xlabel='Degree coefficient of variation',ylabel='(B − A) / max(A, 1)',title='Repaired A advantage and degree heterogeneity')
    axes[1].axhline(0,color='grey',lw=1);axes[1].axvline(0,color='grey',lw=1)
    axes[1].set(xlabel='A benefit over B',ylabel='Noise-2 benefit over B',title='Benefits with a common B baseline')
    for arm in ['A','A_nodrop','B','B_noise2','C','D']:
        curve=mv[mv.arm==arm].groupby(['graph','epoch']).frac_pmax_099.mean().groupby('epoch').agg(['median','count'])
        curve=curve[curve['count']==68]
        axes[2].plot(curve.index,curve['median'],label=arm)
    axes[2].set(xscale='log',xlabel='Step',ylabel='Fraction pmax ≥ .99',title='Confidence trajectories');axes[2].legend(fontsize=8)
    axes[0].legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(OUT/'exploratory_overview.png',dpi=160);plt.close(fig)
    print('ANALYSIS_READY',flush=True)
    print(repair.round(3).to_string(index=False),flush=True)
    print('TOP_REPAIRED_A_STRUCTURE',assoc[(assoc.metric=='repaired')&(assoc.arm=='B')].sort_values('q_global').round(4).to_string(index=False),flush=True)
    print('NOISE_OVERLAP',overlap.round(4).to_string(index=False),flush=True)
    print('TRAJECTORIES',tsummary.to_string(index=False),flush=True)
    manifest={'date':'2026-10-14','n_graphs':68,'n_seeds':5,'arms':ARMS,'endpoint_records':len(ep),
              'structure_tests':len(assoc),'within_family_permutations':4999,
              'versions':{'numpy':np.__version__,'pandas':pd.__version__,'networkx':nx.__version__}}
    (OUT/'analysis_metadata.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__': main()
