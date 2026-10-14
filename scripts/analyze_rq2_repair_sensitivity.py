"""Inspect actual saved logits; reproduce descent and test ten new tie-break seeds.

Trace uses the same steepest strict rule and sorted seeded ties as endpoint.py,
with vectorized gain scans. Original counts/moves and local optimality are checked.
Only separate exploratory outputs are written.
"""
from pathlib import Path
import sys, time, json
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_rq2_exploratory import ROOT, OUT, ARMS, graph_load, bh, corr

torch.set_num_threads(1)
TIE_SEEDS=list(range(10000,10010))

def load_z(root, graph, seed, epoch):
    p=root/f'{graph}_seed{seed}'/'checkpoints'/f'ckpt_{epoch:08d}.pt'
    if not p.exists(): return None
    d=torch.load(p,map_location='cpu',weights_only=True)
    assert d['epoch']==epoch
    return d['Z'].numpy()

def decode(z):
    # Preserve the original float32-softmax argmax convention. Near uniformity,
    # logit differences can disappear on softmax rounding and change tie labels.
    return torch.softmax(torch.from_numpy(z),dim=1).argmax(1).numpy()

def counts(c, neighbours,k):
    return np.array([np.bincount(c[nb],minlength=k) for nb in neighbours],dtype=np.int64)

def descend(c0,neighbours,k,seed):
    c=c0.copy();n=len(c);M=counts(c,neighbours,k); ids=np.arange(n)
    own=M[ids,c];gains=own-M.min(1);initial=gains>0
    rng=np.random.default_rng(seed);seen=np.zeros(n,dtype=bool);moves=0;outside=0
    while True:
        best=gains.max()
        if best<=0:break
        v=int(rng.choice(np.flatnonzero(gains==best)))
        new=int(rng.choice(np.flatnonzero(M[v]==M[v].min())))
        old=int(c[v]);c[v]=new
        nb=neighbours[v];M[nb,old]-=1;M[nb,new]+=1
        touched=np.append(nb,v)
        gains[touched]=M[touched,c[touched]]-M[touched].min(1)
        seen[v]=True;moves+=1;outside+=int(not initial[v])
    assert (M[ids,c]-M.min(1)).max()<=0
    return dict(conflicts=int(M[ids,c].sum()//2),moves=moves,unique_recoloured=int(seen.sum()),
        newly_improvable_recoloured=int((seen&~initial).sum()),moves_outside_initial=outside,
        initial_improvable=int(initial.sum()),repeated_moves=moves-int(seen.sum())),c

def partition_agreement(a,b):
    # Best color-permutation alignment. Avoid confusing label swaps with structural changes.
    _,ia=np.unique(a,return_inverse=True);_,ib=np.unique(b,return_inverse=True)
    cont=np.zeros((ia.max()+1,ib.max()+1),dtype=np.int64)
    np.add.at(cont,(ia,ib),1)
    r,c=linear_sum_assignment(cont,maximize=True)
    return float(cont[r,c].sum()/len(a))

def gini(x):
    if x.sum()==0:return 0.
    x=np.sort(x);n=len(x)
    return float(2*np.dot(np.arange(1,n+1),x)/(n*x.sum())-(n+1)/n)

def main():
    ep=pd.read_csv(OUT/'endpoint_run_metrics.csv')
    features=pd.read_csv(OUT/'graph_features.csv')
    rows=[];ties=[];switches=[];t0=time.time()
    for j,fr in features.iterrows():
        name=fr.graph;k=int(fr.k);g=graph_load(name);n=int(fr.n);m=int(fr.m)
        nb=[np.array(sorted(g[v]),dtype=np.int64) for v in range(n)]
        degree=np.array([len(x) for x in nb]);u,v=np.array(list(g.edges())).T
        for arm in ARMS:
            root=ROOT/('results/main' if arm=='A' else 'results/exp3/'+arm)
            for seed in range(5):
                base=ep[(ep.arm==arm)&(ep.graph==name)&(ep.seed==seed)].iloc[0]
                z=load_z(root,name,seed,100000);assert z is not None
                c=decode(z);own=np.array([(c[nei]==c[i]).sum() for i,nei in enumerate(nb)])
                raw=int(own.sum()//2);assert raw==base.conflicts,(arm,name,seed,'raw')
                report,repaired=descend(c,nb,k,seed)
                assert report['conflicts']==base.conflicts_after_descent,(arm,name,seed,'repaired')
                assert report['moves']==base.descent_moves,(arm,name,seed,'moves')
                assert report['initial_improvable']==base.n_improvable,(arm,name,seed,'improvable')
                entry=dict(arm=arm,graph=name,seed=seed,family=fr.family,family_group=fr.family_group,n=n,m=m,k=k,
                    raw=raw,repaired=report['conflicts'],removed=raw-report['conflicts'],
                    conflicted_vertex_fraction=float((own>0).mean()),conflict_gini=gini(own),
                    conflict_top10_share=float(np.sort(own)[-max(1,int(np.ceil(.1*n))):].sum()/own.sum()) if own.sum() else 0.,
                    max_vertex_conflicts=int(own.max()),**{key:val for key,val in report.items() if key!='conflicts'})
                entry['initial_improvable_per_conflicted']=report['initial_improvable']/max(int((own>0).sum()),1)
                colour_counts=np.bincount(c,minlength=k)
                entry['max_colour_fraction']=float(colour_counts.max()/n)
                entry['decoded_colours_used']=int((colour_counts>0).sum())
                entry['collapsed_one_colour']=bool((colour_counts>0).sum()==1)
                zz=z.astype(np.float64);prob=np.exp(zz-zz.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
                confident=prob.max(1)>=.99
                initialM=counts(c,nb,k);improvable=initialM[np.arange(n),c]>initialM.min(1)
                entry['confident_improvable_vertices']=int((confident&improvable).sum())
                entry['improvable_vertices']=int(improvable.sum())
                entry['confident_improvable_fraction']=float(confident[improvable].mean()) if improvable.any() else np.nan
                entry['unique_recoloured_fraction']=report['unique_recoloured']/n
                entry['newly_improvable_fraction']=report['newly_improvable_recoloured']/max(report['unique_recoloured'],1)
                entry['repeated_move_fraction']=report['repeated_moves']/max(report['moves'],1)
                # Ten independent seeds, no best-of selection; deterministic local optima need no rerun.
                outcomes=[];agreements=[]
                for ts in TIE_SEEDS:
                    if report['moves']==0: other=report;col=repaired
                    else: other,col=descend(c,nb,k,ts)
                    outcomes.append(other['conflicts']);agreements.append(partition_agreement(repaired,col))
                    ties.append(dict(arm=arm,graph=name,run_seed=seed,tie_seed=ts,
                        conflicts=other['conflicts'],moves=other['moves'],aligned_partition_agreement=agreements[-1]))
                entry.update(tie_mean=np.mean(outcomes),tie_min=min(outcomes),tie_max=max(outcomes),
                    tie_range=max(outcomes)-min(outcomes),tie_std=np.std(outcomes),
                    mean_partition_agreement=np.mean(agreements),tie_count_varies=max(outcomes)>min(outcomes))
                rows.append(entry)
                # Assignment changes between common checkpoints; not inferred from confidence.
                previous=None
                for epoch in [100,1000,10000,50000,100000]:
                    zz=z if epoch==100000 else load_z(root,name,seed,epoch)
                    if zz is None:continue
                    cc=decode(zz)
                    if previous is not None:
                        pe,pc=previous
                        switches.append(dict(arm=arm,graph=name,seed=seed,family_group=fr.family_group,
                            from_step=pe,to_step=epoch,vertex_switch_fraction=float((cc!=pc).mean()),
                            partition_switch_fraction=1-partition_agreement(cc,pc),
                            edge_conflict_status_switch=float(((cc[u]==cc[v])!=(pc[u]==pc[v])).mean())))
                    previous=(epoch,cc)
        pd.DataFrame(rows).to_csv(OUT/'repair_trace_runs.csv',index=False)
        pd.DataFrame(ties).to_csv(OUT/'repair_tie_runs.csv',index=False)
        pd.DataFrame(switches).to_csv(OUT/'assignment_switch_runs.csv',index=False)
        print(f'graph {j+1}/68 {name} elapsed {time.time()-t0:.1f}s',flush=True)
    rr=pd.DataFrame(rows); tt=pd.DataFrame(ties); ss=pd.DataFrame(switches)
    assert len(rr)==3060 and len(tt)==30600 and rr.removed.ge(0).all()
    pg=rr.groupby(['arm','graph','family','family_group']).mean(numeric_only=True).reset_index()
    pg.to_csv(OUT/'repair_trace_graphs.csv',index=False)
    summary=rr.groupby('arm').agg(runs=('graph','size'),variable_tie_runs=('tie_count_varies','sum'),
        mean_tie_range=('tie_range','mean'),mean_tie_conflicts=('tie_mean','mean'),
        mean_original_conflicts=('repaired','mean'),mean_newly_improvable_recoloured=('newly_improvable_recoloured','mean'),
        mean_repeated_moves=('repeated_moves','mean')).reset_index()
    summary.to_csv(OUT/'repair_sensitivity_summary.csv',index=False)
    rfamily=pg.groupby(['family_group','arm']).median(numeric_only=True).reset_index()
    rfamily.to_csv(OUT/'repair_trace_by_family.csv',index=False)
    # Original versus tie-averaged rankings, using the original practical-tie rule.
    rank=[]
    for metric in ['repaired','tie_mean']:
        w=pg.pivot(index='graph',columns='arm',values=metric)
        for x,y in [('B','A'),('C','A'),('D','A'),('B_noise2','A'),('A_nodrop','A')]:
            d=(w[x]-w[y])/np.maximum(w[y],1)
            tie=(abs(w[x]-w[y])<1)|(abs(d)<=.05)
            verdict=np.where(tie,'tie',np.where(w[x]<w[y],'X better','Y better'))
            for graph,di,vi in zip(w.index,d,verdict):
                rank.append(dict(graph=graph,metric=metric,X=x,Y=y,d=di,verdict=vi))
    rank=pd.DataFrame(rank).merge(features[['graph','family_group']],on='graph')
    rank.to_csv(OUT/'repair_sensitivity_verdicts.csv',index=False)
    rank.groupby(['metric','X','Y']).agg(X_better=('verdict',lambda x:(x=='X better').sum()),
        ties=('verdict',lambda x:(x=='tie').sum()),Y_better=('verdict',lambda x:(x=='Y better').sum()),
        median_d=('d','median')).reset_index().to_csv(OUT/'repair_sensitivity_comparisons.csv',index=False)
    changes=rank.pivot(index=['graph','X','Y'],columns='metric',values='verdict')
    changes['changed']=changes.repaired!=changes.tie_mean
    changes.to_csv(OUT/'repair_sensitivity_changed_graphs.csv')
    # Graph-level descriptive associations; raw concentration and final repair gain have scale/confounding.
    ass=[]
    for arm in ARMS:
        df=pg[pg.arm==arm]
        for c in ['conflict_top10_share','conflict_gini','conflicted_vertex_fraction',
                  'initial_improvable_per_conflicted','newly_improvable_fraction','repeated_move_fraction','tie_range']:
            for target in ['repair_fraction','repaired_per_edge']:
                y=((df.raw-df.repaired)/df.raw.clip(lower=1)) if target=='repair_fraction' else df.repaired/df.m
                rho,p=corr(df[c].to_numpy(),y.to_numpy())
                ass.append(dict(arm=arm,diagnostic=c,target=target,rho=rho,p=p))
    ass=pd.DataFrame(ass);ass['q']=bh(ass.p);ass.to_csv(OUT/'repair_trace_correlations.csv',index=False)
    sg=ss.groupby(['arm','graph','family_group','from_step','to_step']).mean(numeric_only=True).reset_index()
    sg.to_csv(OUT/'assignment_switch_graphs.csv',index=False)
    sg.groupby(['arm','from_step','to_step']).median(numeric_only=True).reset_index().to_csv(OUT/'assignment_switch_summary.csv',index=False)
    sg.groupby(['family_group','arm','from_step','to_step']).median(numeric_only=True).reset_index().to_csv(OUT/'assignment_switch_by_family.csv',index=False)
    print('REPAIR_READY',flush=True);print(summary.round(3).to_string(index=False),flush=True)

if __name__=='__main__':main()
