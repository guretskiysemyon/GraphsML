"""Robustness and family-aware diagnostics for the exploratory RQ2 appendices."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_rq2_exploratory import ROOT,OUT,FEATURES,ARMS,corr,bh,group_family

def residual_rho(x,y,families):
    mask=np.isfinite(x)&np.isfinite(y)
    x=rankdata(np.asarray(x)[mask]);y=rankdata(np.asarray(y)[mask]);fam=np.asarray(families)[mask]
    for f in np.unique(fam):
        ids=fam==f;x[ids]-=x[ids].mean();y[ids]-=y[ids].mean()
    norm=np.linalg.norm(x)*np.linalg.norm(y)
    if norm==0:return np.nan,np.nan,int(mask.sum())
    rho=float(x@y/norm);rng=np.random.default_rng(20261014);perms=np.tile(np.arange(len(x)),(4999,1))
    for f in np.unique(fam):
        ids=np.flatnonzero(fam==f)
        perms[:,ids]=np.array([rng.permutation(ids) for _ in range(4999)])
    pr=(y[perms]@x)/norm
    return rho,float((1+(abs(pr)>=abs(rho)-1e-12).sum())/5000),int(mask.sum())

def partial_rank(x,y,controls):
    xx=rankdata(x);yy=rankdata(y)
    c=np.column_stack([np.ones(len(xx))]+[rankdata(v) for v in controls])
    xr=xx-c@np.linalg.lstsq(c,xx,rcond=None)[0];yr=yy-c@np.linalg.lstsq(c,yy,rcond=None)[0]
    den=np.linalg.norm(xr)*np.linalg.norm(yr)
    return float(xr@yr/den) if den>1e-9 else np.nan

def main():
    f=pd.read_csv(OUT/'graph_features.csv').set_index('graph')
    comp=pd.read_csv(OUT/'paired_effects.csv')
    # Sensitivity for A versus plain B; denominators and easy endpoint floors.
    e=comp[(comp.metric=='repaired')&(comp.X=='B')&(comp.Y=='A')].set_index('graph').loc[f.index]
    robustness=[]
    target_defs={'relative_to_A':e.d,'symmetric_relative':(e.X_value-e.Y_value)/np.maximum(np.maximum(e.X_value,e.Y_value),1),
                 'edge_normalised_difference':e.edge_delta}
    for c in FEATURES:
        for outcome,y in target_defs.items():
            rho,p=corr(f[c].to_numpy(),y.to_numpy())
            robustness.append(dict(feature=c,check=outcome,n=len(f),rho=rho,p=p))
        mask=np.maximum(e.X_value,e.Y_value)>=5
        rho,p=corr(f.loc[mask,c].to_numpy(),e.loc[mask,'d'].to_numpy())
        robustness.append(dict(feature=c,check='exclude_both_endpoints_below_5',n=int(mask.sum()),rho=rho,p=p))
        for fam in sorted(f.family_group.unique()):
            keep=f.family_group!=fam;rho,p=corr(f.loc[keep,c].to_numpy(),e.loc[keep,'d'].to_numpy())
            robustness.append(dict(feature=c,check='exclude_group_'+fam,n=int(keep.sum()),rho=rho,p=p))
        controls=[f[cov].to_numpy() for cov in ['n','k','density'] if cov!=c]
        robustness.append(dict(feature=c,check='partial_rank_control_n_k_density_except_self',n=68,
            rho=partial_rank(f[c].to_numpy(),e.d.to_numpy(),controls),p=np.nan))
    pd.DataFrame(robustness).to_csv(OUT/'structure_robustness.csv',index=False)
    # Quantile groups describe continuous predictors without choosing thresholds to maximize wins.
    bins=[]
    for c in ['n','density','degree_cv','clique_lb_over_k','mean_degree_over_k']:
        # Tied feature values remain together; qcut may produce fewer than three bins.
        bb=pd.qcut(f[c],3,duplicates='drop')
        for label,ids in bb.groupby(bb,observed=True).groups.items():
            for x,y in [('B','A'),('C','A'),('D','A'),('B_noise2','A'),('B_noise2','B')]:
                for metric in ['raw','repaired']:
                    df=comp[(comp.graph.isin(ids))&(comp.X==x)&(comp.Y==y)&(comp.metric==metric)]
                    bins.append(dict(feature=c,bin=str(label),feature_min=f.loc[ids,c].min(),feature_max=f.loc[ids,c].max(),
                        graphs=len(df),X=x,Y=y,metric=metric,X_better=(df.verdict=='X better').sum(),
                        ties=(df.verdict=='tie').sum(),Y_better=(df.verdict=='Y better').sum(),median_d=df.d.median()))
    pd.DataFrame(bins).to_csv(OUT/'structural_bins.csv',index=False)
    # Normalise color usage to palette size to avoid interpreting k as a behavior diagnostic.
    mv=pd.read_csv(ROOT/'results/analysis/exp3_movement_tidy.csv');mv=mv[mv.epoch==100000]
    dg=mv.groupby(['arm','graph']).mean(numeric_only=True).reset_index().merge(f.reset_index()[['graph','family','family_group','k']],on='graph')
    ep=pd.read_csv(OUT/'endpoint_graph_metrics.csv')
    dg=dg.merge(ep[['arm','graph','conflicts_after_descent','repair_fraction']],on=['arm','graph'])
    dg['colours_used_fraction']=dg.n_colours_used/dg.k
    dg['repaired_per_edge']=dg.conflicts_after_descent/dg.m
    dg['entropy_normalised']=dg.eval_entropy/np.log(dg.k)
    tg=pd.read_csv(OUT/'trajectory_run_metrics.csv').groupby(['arm','graph']).mean(numeric_only=True).reset_index()
    dg=dg.merge(tg[['arm','graph','confidence100','first_confident_step','normalised_late_raw_range']],on=['arm','graph'])
    di=[]
    for arm in ARMS:
        rr=dg[dg.arm==arm]
        for c in ['margin_median','frac_pmax_099','entropy_normalised','logit_dist_centred_rel','colours_used_fraction',
                  'confidence100','first_confident_step','normalised_late_raw_range']:
            for target in ['repair_fraction','repaired_per_edge']:
                x=rr[c].to_numpy();y=rr[target].to_numpy();rho,p=corr(x,y)
                wr,wp,n=residual_rho(x,y,rr.family.to_numpy())
                mask=np.isfinite(x)&np.isfinite(y)
                controls=[f.loc[rr.graph,cov].to_numpy()[mask] for cov in ['n','k','density']]
                partial=partial_rank(x[mask],y[mask],controls)
                di.append(dict(arm=arm,diagnostic=c,target=target,n=n,rho=rho,p=p,within_family_rho=wr,within_family_p=wp,
                    partial_rho_n_k_density=partial))
    di=pd.DataFrame(di);di['q']=bh(di.p);di['q_within_family']=bh(di.within_family_p)
    di.to_csv(OUT/'diagnostics_family_adjusted.csv',index=False)
    dg.to_csv(OUT/'diagnostics_graph_metrics.csv',index=False)
    # Benefits with within-family and denominator sensitivity. Correlations are not causal evidence.
    w=ep.pivot(index='graph',columns='arm',values='conflicts_after_descent').loc[f.index]
    no=[]
    for noise in ['B_noise1','B_noise2']:
        for norm in ['B','edges']:
            denom=np.maximum(w.B,1) if norm=='B' else f.m
            a=(w.B-w.A)/denom;b=(w.B-w[noise])/denom
            rho,p=corr(a.to_numpy(),b.to_numpy());wr,wp,n=residual_rho(a.to_numpy(),b.to_numpy(),f.family.to_numpy())
            no.append(dict(noise=noise,normalisation=norm,n=n,rho=rho,p=p,within_family_rho=wr,within_family_p=wp))
    pd.DataFrame(no).to_csv(OUT/'noise_overlap_robustness.csv',index=False)
    # The 13 genuinely extended graphs, compared with themselves at both budgets.
    tidy=pd.read_csv(ROOT/'results/analysis/exp3_tidy.csv')
    afinal=tidy[tidy.arm=='A'].groupby('graph').epoch.max();extended=afinal[afinal==300000].index
    assert len(extended)==13
    ext=tidy[(tidy.graph.isin(extended))&tidy.epoch.isin([100000,300000])]
    ext=ext.merge(f.reset_index()[['graph','family','family_group','m']],on='graph')
    eg=ext.groupby(['arm','graph','epoch']).agg(raw=('conflicts','mean'),repaired=('conflicts_after_descent','mean'),seeds=('seed','nunique')).reset_index()
    er=[]
    for arm in ARMS:
        rr=eg[eg.arm==arm];w2=rr.pivot(index='graph',columns='epoch',values=['raw','repaired'])
        if 300000 not in w2['raw'].columns:continue
        complete=rr.pivot(index='graph',columns='epoch',values='seeds')
        ids=complete.index[complete.eq(5).all(axis=1)]
        valid=w2.loc[ids].dropna()
        if len(valid)==0:continue
        for metric in ['raw','repaired']:
            start=valid[metric][100000];end=valid[metric][300000]
            delta=end-start;d=delta/np.maximum(start,1);tie=(abs(delta)<1)|(abs(d)<=.05)
            er.append(dict(arm=arm,metric=metric,graphs=len(valid),better_at_300k=int((~tie&(delta<0)).sum()),
                ties=int(tie.sum()),worse_at_300k=int((~tie&(delta>0)).sum()),mean100k=start.mean(),mean300k=end.mean(),median_d=d.median()))
    eg.to_csv(OUT/'extended_subset_graphs.csv',index=False);pd.DataFrame(er).to_csv(OUT/'extended_subset_summary.csv',index=False)
    print('ROBUSTNESS_READY',flush=True)
    print(pd.DataFrame(robustness)[pd.DataFrame(robustness).feature.isin(['degree_cv','density'])].round(4).to_string(index=False),flush=True)
    print(pd.DataFrame(no).round(4).to_string(index=False),flush=True)
    print(pd.DataFrame(er).round(3).to_string(index=False),flush=True)

if __name__=='__main__':main()
