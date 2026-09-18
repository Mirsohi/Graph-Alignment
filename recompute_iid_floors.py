from pathlib import Path
import sys, json, argparse
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
from exact_posterior_benchmark import exact_posterior_auto, generate_paired_instances, likelihood_beta
from iid_initialization_analysis import iid_row_tv_replicates

def main():
 p=argparse.ArgumentParser(); p.add_argument('--budget-raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reps',type=int,default=500);a=p.parse_args()
 raw=pd.read_csv(a.budget_raw); rows=[]
 for n in sorted(raw.n.unique()):
  for seed in sorted(raw[raw.n==n].graph_seed.unique()):
   inst=generate_paired_instances(int(n),.3,[.1],int(seed))[0]; exact=exact_posterior_auto(inst,likelihood_beta(.3,.1))
   for _,r in raw[(raw.n==n)&(raw.graph_seed==seed)].iterrows():
    method=str(r.method); budget=int(r.local_proposals_per_chain); retained=int(r.retained_samples_total)
    rs=310_000_000+int(n)*1_000_000+int(seed)*10_000+budget+(0 if method=='local' else 1)
    vals=iid_row_tv_replicates(exact.marginals,retained,a.reps,np.random.default_rng(rs))
    rows.append(dict(n=int(n),graph_seed=int(seed),method=method,local_proposals_per_chain=budget,retained_samples_total=retained,iid_replicates=a.reps,iid_floor_mean=float(vals.mean()),iid_floor_sd=float(vals.std(ddof=1)),iid_floor_q025=float(np.quantile(vals,.025)),iid_floor_q975=float(np.quantile(vals,.975)),observed_row_tv=float(r.mean_row_tv),excess_row_tv=float(r.mean_row_tv-vals.mean()),observed_over_floor_ratio=float(r.mean_row_tv/vals.mean())))
   print('floor',n,seed,flush=True)
 d=pd.DataFrame(rows); a.output.mkdir(parents=True,exist_ok=True);d.to_csv(a.output/'iid_floor_raw.csv',index=False)
 s=d.groupby(['n','method','local_proposals_per_chain']).agg(instances=('graph_seed','nunique'),observed_row_tv_mean=('observed_row_tv','mean'),observed_row_tv_sem=('observed_row_tv','sem'),iid_floor_mean=('iid_floor_mean','mean'),iid_floor_sem=('iid_floor_mean','sem'),excess_row_tv_mean=('excess_row_tv','mean'),excess_row_tv_sem=('excess_row_tv','sem'),mean_observed_over_floor_ratio=('observed_over_floor_ratio','mean')).reset_index();s.to_csv(a.output/'iid_floor_summary.csv',index=False)
 (a.output/'floor_metadata.json').write_text(json.dumps({'reps':a.reps,'note':'row-wise multinomial simulation gives exact expected row-TV; quantiles ignore cross-row dependence and are descriptive only'},indent=2))
if __name__=='__main__':main()
