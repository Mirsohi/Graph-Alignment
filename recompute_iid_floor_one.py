from pathlib import Path
import sys,argparse
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from exact_posterior_benchmark import exact_posterior_auto,generate_paired_instances,likelihood_beta
from iid_initialization_analysis import iid_row_tv_replicates
p=argparse.ArgumentParser();p.add_argument('--budget-raw',type=Path,required=True);p.add_argument('--n',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--reps',type=int,default=500);a=p.parse_args()
raw=pd.read_csv(a.budget_raw); raw=raw[(raw.n==a.n)&(raw.graph_seed==a.seed)]
inst=generate_paired_instances(a.n,.3,[.1],a.seed)[0];ex=exact_posterior_auto(inst,likelihood_beta(.3,.1));rows=[]
for _,r in raw.iterrows():
 method=str(r.method);budget=int(r.local_proposals_per_chain);ret=int(r.retained_samples_total);rs=310_000_000+a.n*1_000_000+a.seed*10_000+budget+(0 if method=='local' else 1);v=iid_row_tv_replicates(ex.marginals,ret,a.reps,np.random.default_rng(rs));rows.append(dict(n=a.n,graph_seed=a.seed,method=method,local_proposals_per_chain=budget,retained_samples_total=ret,iid_replicates=a.reps,iid_floor_mean=v.mean(),iid_floor_sd=v.std(ddof=1),iid_floor_q025=np.quantile(v,.025),iid_floor_q975=np.quantile(v,.975),observed_row_tv=r.mean_row_tv,excess_row_tv=r.mean_row_tv-v.mean(),observed_over_floor_ratio=r.mean_row_tv/v.mean()))
a.out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(a.out,index=False)
