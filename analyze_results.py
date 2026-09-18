"""Aggregate the canonical experiment outputs and write summary tables."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results'

def boot_ci(x, seed=1, B=20000):
    x=np.asarray(x,float); rng=np.random.default_rng(seed)
    vals=np.mean(x[rng.integers(0,len(x),size=(B,len(x)))],axis=1)
    return float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

budget=pd.read_csv(RES/'budget_curves'/'budget_curve_raw.csv')
floors=pd.read_csv(RES/'floor_initialization'/'iid_floor_raw.csv')
init=pd.read_csv(RES/'floor_initialization'/'initialization_raw.csv')
smc=pd.read_csv(RES/'inference_baselines'/'inference_baselines_raw.csv')
full=pd.read_csv(RES/'full_tv_n8'/'full_tv_raw.csv')
family=pd.read_csv(RES/'family_replication'/'family_sampler_results.csv')
barker=pd.read_csv(RES/'barker_mechanism'/'barker_flux_raw.csv')

# Main paired results at maximum budget.
key=[]
for n in [8,9,10]:
    d=budget[(budget.n==n)&(budget.local_proposals_per_chain==262144)].pivot(index='graph_seed',columns='method',values='mean_row_tv').dropna()
    diff=d['local']-d['parallel_tempering']; lo,hi=boot_ci(diff,100+n)
    key.append(dict(result='local_minus_pt',n=n,instances=len(diff),estimate=diff.mean(),ci_low=lo,ci_high=hi,p_value=wilcoxon(diff,alternative='two-sided').pvalue,local_mean=d.local.mean(),pt_mean=d.parallel_tempering.mean()))
    s=smc[smc.n==n].set_index('graph_seed').mean_row_tv
    common=sorted(set(d.index)&set(s.index)); ds=d.loc[common,'local']-s.loc[common]; lo2,hi2=boot_ci(ds,200+n)
    key.append(dict(result='local_minus_smc',n=n,instances=len(ds),estimate=ds.mean(),ci_low=lo2,ci_high=hi2,p_value=wilcoxon(ds).pvalue,local_mean=d.loc[common,'local'].mean(),pt_mean=s.loc[common].mean()))
# initialization
for n in [8,9,10]:
    p=init[init.n==n].pivot(index='graph_seed',columns='initialization',values='mean_row_tv').dropna(); diff=p.common_random-p.independent_uniform; lo,hi=boot_ci(diff,300+n)
    key.append(dict(result='common_minus_independent_start',n=n,instances=len(diff),estimate=diff.mean(),ci_low=lo,ci_high=hi,p_value=wilcoxon(diff).pvalue,local_mean=p.common_random.mean(),pt_mean=p.independent_uniform.mean()))
keydf=pd.DataFrame(key); keydf.to_csv(RES/'key_results.csv',index=False)

# Text summary.
maxfloor=floors[floors.local_proposals_per_chain==262144].groupby(['n','method']).agg(obs=('observed_row_tv','mean'),floor=('iid_floor_mean','mean'),excess=('excess_row_tv','mean')).reset_index()
init_summary=init.groupby(['n','initialization']).mean_row_tv.agg(['mean','sem']).reset_index()
smc_summary=smc.groupby('n').mean_row_tv.agg(['mean','sem']).reset_index()
full_summary=full.groupby('method').agg(full_tv=('full_posterior_tv','mean'),floor=('iid_full_tv_mean','mean'),excess=('excess_full_tv','mean')).reset_index()
# family columns inspect flexibly
summary_lines=['# Results','', '## Maximum-budget paired comparisons','',keydf.to_markdown(index=False),'','## I.I.D. floors','',maxfloor.to_markdown(index=False),'','## Initialization','',init_summary.to_markdown(index=False),'','## Population SMC','',smc_summary.to_markdown(index=False),'','## Full posterior TV at n=8','',full_summary.to_markdown(index=False)]
(RES/'RESULTS.md').write_text('\n'.join(summary_lines))
print(keydf.to_string(index=False)); print('\n',maxfloor.to_string(index=False)); print('\n',smc_summary.to_string(index=False))
