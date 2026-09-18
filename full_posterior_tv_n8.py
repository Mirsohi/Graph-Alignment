"""Permutation-level total-variation audit at n=8.

Runs the maximum-budget local and parallel-tempering arms, compares their
empirical distribution over all 8! permutations with the exact posterior, and
reports matched i.i.d. finite-sample floors.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from exact_posterior_benchmark import exact_posterior_auto,generate_paired_instances,likelihood_beta
from budget_curve_experiment import run_local_compiled,run_pt_compiled,initial_local,initial_pt,seed_for,marginals,row_tv,warmup

def empirical_probs(samples, index, K):
    c=np.zeros(K,dtype=np.int64)
    for x in samples: c[index[tuple(int(v) for v in x)]]+=1
    return c/len(samples)

def run(args):
    args.output_dir.mkdir(parents=True,exist_ok=True); warmup(); rows=[]
    for g in range(args.instances):
        inst=generate_paired_instances(8,args.p,[args.noise],g)[0]; beta=likelihood_beta(args.p,args.noise); ex=exact_posterior_auto(inst,beta)
        index={tuple(int(v) for v in p):i for i,p in enumerate(ex.permutations)}
        for code,method in enumerate(['local','parallel_tempering']):
            chains=[]
            for c in range(args.chains):
                seed=seed_for(8,g,code,c)
                if method=='local':
                    s,_,_=run_local_compiled(inst.A,inst.B,beta,args.budget,int(args.budget*.25),5,seed,initial_local(8,seed))
                else:
                    sweeps=args.budget//8
                    s,_,_,_,_=run_pt_compiled(inst.A,inst.B,beta,sweeps,int(sweeps*.25),1,seed,initial_pt(8,8,seed))
                chains.append(s)
            samples=np.vstack(chains); phat=empirical_probs(samples,index,len(ex.probabilities)); fulltv=.5*np.abs(phat-ex.probabilities).sum(); rtv=row_tv(marginals(samples,8),ex.marginals)
            rng=np.random.default_rng(600_000_000+g*100+code); floors=[]
            for _ in range(args.iid_replicates):
                counts=rng.multinomial(len(samples),ex.probabilities); floors.append(.5*np.abs(counts/len(samples)-ex.probabilities).sum())
            rows.append(dict(graph_seed=g,method=method,samples=len(samples),full_posterior_tv=fulltv,iid_full_tv_mean=np.mean(floors),iid_full_tv_sd=np.std(floors,ddof=1),excess_full_tv=fulltv-np.mean(floors),row_tv=rtv))
        print('full TV n8 seed',g,flush=True)
    raw=pd.DataFrame(rows); raw.to_csv(args.output_dir/'full_tv_raw.csv',index=False)
    summary=raw.groupby('method').agg(instances=('graph_seed','size'),mean_full_tv=('full_posterior_tv','mean'),sem_full_tv=('full_posterior_tv','sem'),mean_iid_floor=('iid_full_tv_mean','mean'),mean_excess_full_tv=('excess_full_tv','mean'),mean_row_tv=('row_tv','mean')).reset_index(); summary.to_csv(args.output_dir/'full_tv_summary.csv',index=False)
    json.dump(vars(args)|{'output_dir':str(args.output_dir)},open(args.output_dir/'metadata.json','w'),indent=2); print(summary.to_string(index=False))
def parse():
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,default=ROOT/'results'/'full_tv_n8'); p.add_argument('--instances',type=int,default=20); p.add_argument('--p',type=float,default=.3); p.add_argument('--noise',type=float,default=.1); p.add_argument('--budget',type=int,default=262144); p.add_argument('--chains',type=int,default=4); p.add_argument('--iid-replicates',type=int,default=100); return p.parse_args()
if __name__=='__main__': run(parse())
