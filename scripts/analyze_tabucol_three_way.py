"""Compare raw and repaired PI-GNN at 100k with the existing TabuCol reference.

Run from repository root. TabuCol budget is training-only, not training + descent.
Outputs graph/family summaries, a section-ready table, and PNG/SVG figures.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/rq43-matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis'
t = pd.read_csv(OUT/'tabucol_per_graph.csv').set_index('graph')
d = pd.read_csv(OUT/'tidy_seed_level.csv').query('epoch == 100000')
assert len(d)==340 and d.graph.nunique()==68
assert d.groupby('graph').size().eq(5).all()
p = d.groupby('graph').agg(raw=('conflicts','mean'), repaired=('conflicts_after_descent','mean'))
g = t[['family','tabu_mean','budget_s','tabu_seeds','pignn_mean']].join(p,validate='one_to_one')
assert len(g)==68 and g.notna().all().all() and g.tabu_seeds.eq(10).all()
assert np.allclose(g.repaired,g.pignn_mean)
assert (g.repaired<=g.raw).all()
for col in ['raw','repaired']:
    delta=g[col]-g.tabu_mean
    rel=delta/g.tabu_mean.clip(lower=1)
    tie=(delta.abs()<1)|(rel.abs()<=.05)
    g[col+'_verdict']=np.where(tie,'tie',np.where(delta>0,'tabucol_fewer','pignn_fewer'))
    g[col+'_ratio']=g[col]/g.tabu_mean.clip(lower=1)
g.to_csv(OUT/'tabucol_three_way_per_graph.csv')
f=g.groupby('family').agg(graphs=('raw','size'),tabucol=('tabu_mean','mean'),
    raw=('raw','mean'),repaired=('repaired','mean'),
    raw_ratio=('raw_ratio','median'),repaired_ratio=('repaired_ratio','median'))
f.to_csv(OUT/'tabucol_three_way_by_family.csv')
rows=[]
for col,label in [('raw','PI-GNN (raw argmax)'),('repaired','PI-GNN + strict descent')]:
    counts=g[col+'_verdict'].value_counts()
    rows.append({'Output':label,'Mean conflicts':g[col].mean(),
                 'TabuCol fewer':int(counts.get('tabucol_fewer',0)),
                 'Practical ties':int(counts.get('tie',0)),
                 'PI-GNN fewer':int(counts.get('pignn_fewer',0))})
summary=pd.DataFrame(rows)
summary.to_csv(OUT/'tabucol_three_way_summary.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
    'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
fig=plt.figure(figsize=(7.6,6.6))
gs=fig.add_gridspec(2,2,height_ratios=[1,1.15],hspace=.62,wspace=.36)
blue,orange='#0072B2','#D55E00'
lo=min(g[['raw','repaired','tabu_mean']].min())*.7
hi=max(g[['raw','repaired','tabu_mean']].max())*1.4
for ax,col,color,title in [(fig.add_subplot(gs[0,0]),'raw',blue,'(a) Raw PI-GNN'),
                          (fig.add_subplot(gs[0,1]),'repaired',orange,'(b) PI-GNN + descent')]:
    ax.plot([lo,hi],[lo,hi],color='0.55',ls='--',lw=.8)
    ax.scatter(g.tabu_mean,g[col],s=20,facecolors='none',edgecolors=color,linewidths=.9)
    ax.set(xscale='log',yscale='log',xlim=(lo,hi),ylim=(lo,hi),
           xlabel='TabuCol conflicts',ylabel='PI-GNN conflicts',title=title)
    ax.grid(color='0.92',lw=.5)
    counts=g[col+'_verdict'].value_counts()
    ax.text(.04,.96,f"TabuCol fewer / tie / PI-GNN fewer\n{counts.get('tabucol_fewer',0)} / {counts.get('tie',0)} / {counts.get('pignn_fewer',0)}",
            transform=ax.transAxes,va='top',fontsize=8)
ax=fig.add_subplot(gs[1,:])
f=f.sort_values('repaired_ratio')
y=np.arange(len(f))
for i,row in enumerate(f.itertuples()):
    ax.plot([row.raw_ratio,row.repaired_ratio],[i,i],color='0.75',lw=1,zorder=1)
ax.scatter(f.raw_ratio,y,s=26,marker='o',facecolors='none',edgecolors=blue,label='Raw PI-GNN',zorder=3)
ax.scatter(f.repaired_ratio,y,s=26,marker='s',color=orange,label='PI-GNN + descent',zorder=3)
ax.axvline(1,color='0.4',ls='--',lw=.8)
ax.set(xscale='log',yticks=y,yticklabels=[f'{name} (n={int(row.graphs)})' for name,row in f.iterrows()],
       xlabel='Median graph-level PI-GNN / TabuCol conflict ratio (log scale)',
       title='(c) Family comparison · ratio > 1 favors TabuCol')
ax.invert_yaxis()
ax.grid(axis='x',color='0.92',lw=.5)
ax.legend(loc='upper right',frameon=False,fontsize=8)
fig.subplots_adjust(left=.17,right=.98,top=.95,bottom=.09)
for ext in ['png','svg']:
    fig.savefig(OUT/f'tabucol_three_way.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
print(summary.to_string(index=False))
print('TabuCol mean:',g.tabu_mean.mean())
print('Descent improves graph means:',int((g.repaired<g.raw).sum()))
print('Raw PI-GNN wins by family:',g.loc[g.raw_verdict.eq('pignn_fewer')].groupby('family').size().to_dict())
print(f.to_string())
