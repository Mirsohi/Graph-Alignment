"""Mechanism audit for random and Barker locally-balanced transpositions.

On certified rigid near-twin graphs, evaluate the proposal mass and accepted
one-step probability flux toward the competing block orientation along a
canonical transposition path. This converts the surprising Barker ablation into
a directly checkable mechanism result. The script also numerically verifies
detailed balance on a small instance.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code')); from near_twin_family_experiment import qualifying_graphs  # noqa

@njit(cache=True)
def delta(G,pi,u,v):
    a=pi[u]; b=pi[v]; d=0; n=len(pi)
    for k in range(n):
        if k==u or k==v: continue
        pk=pi[k]
        d += (G[u,k]-G[v,k])*(G[b,pk]-G[a,pk])
    return d

@njit(cache=True)
def score(G,pi):
    s=0; n=len(pi)
    for i in range(n):
        for j in range(i+1,n):
            if G[i,j] and G[pi[i],pi[j]]: s+=1
    return s

@njit(cache=True)
def rcoord(pi,m):
    r=0
    for i in range(m):
        if pi[i]>=m: r+=1
    return r

@njit(cache=True)
def sigmoid(x):
    if x>=0:
        z=math.exp(-x); return 1.0/(1.0+z)
    z=math.exp(x); return z/(1.0+z)

@njit(cache=True)
def all_pairs(n):
    D=n*(n-1)//2
    us=np.empty(D,np.int16); vs=np.empty(D,np.int16); t=0
    for u in range(n):
        for v in range(u+1,n):
            us[t]=u; vs[t]=v; t+=1
    return us,vs

@njit(cache=True)
def normalizer(G,pi,beta,us,vs):
    z=0.0
    for t in range(len(us)):
        z += sigmoid(beta*delta(G,pi,int(us[t]),int(vs[t])))
    return z

@njit(cache=True)
def flux_at_state(G,pi,m,beta):
    n=len(pi); us,vs=all_pairs(n); D=len(us); r0=rcoord(pi,m)
    ds=np.empty(D,np.int16); weights=np.empty(D,np.float64); z=0.0
    for t in range(D):
        d=delta(G,pi,int(us[t]),int(vs[t])); ds[t]=d
        w=sigmoid(beta*d); weights[t]=w; z+=w
    rand_prop=0.0; rand_flux=0.0; bark_prop=0.0; bark_flux=0.0; progress=0
    for t in range(D):
        u=int(us[t]); v=int(vs[t])
        tmp=pi[u]; pi[u]=pi[v]; pi[v]=tmp
        is_progress=rcoord(pi,m)>r0
        if is_progress:
            progress+=1
            rand_prop += 1.0/D
            a=1.0 if ds[t]>=0 else math.exp(beta*ds[t])
            rand_flux += (1.0/D)*a
            q=weights[t]/z; bark_prop += q
            zp=normalizer(G,pi,beta,us,vs)
            acc=1.0 if z<=zp else z/zp
            bark_flux += q*acc
        tmp=pi[u]; pi[u]=pi[v]; pi[v]=tmp
    return progress,rand_prop,rand_flux,bark_prop,bark_flux

def canonical_states(m):
    pi=np.arange(2*m,dtype=np.int16); states=[pi.copy()]
    for i in range(m):
        pi[i],pi[m+i]=pi[m+i],pi[i]
        states.append(pi.copy())
    return states

def validate_detailed_balance(G,beta):
    n=len(G); pi=np.arange(n,dtype=np.int16); us,vs=all_pairs(n); z=normalizer(G,pi,beta,us,vs); s=score(G,pi)
    worst=0.0
    for t in range(len(us)):
        u=int(us[t]);v=int(vs[t]);d=delta(G,pi,u,v); w=sigmoid(beta*d); q=w/z
        prop=pi.copy();prop[u],prop[v]=prop[v],prop[u]
        zp=normalizer(G,prop,beta,us,vs); acc=min(1.0,z/zp)
        # reverse Barker weight = sigmoid(-beta*d)
        qr=sigmoid(-beta*d)/zp; accr=min(1.0,zp/z)
        left=math.exp(beta*s)*q*acc; right=math.exp(beta*(s+d))*qr*accr
        worst=max(worst,abs(left-right)/max(left,right,1e-300))
    return worst

def run(args):
    args.output_dir.mkdir(parents=True,exist_ok=True); rows=[]; cert=[]
    firstG=None
    for m in args.m_values:
        graphs=qualifying_graphs(m,args.instances_per_size,args.bridge_vertex)
        for g in graphs:
            G=g['G'].astype(np.int8); firstG=G if firstG is None else firstG
            for r,pi in enumerate(canonical_states(m)):
                if r==m: continue
                progress,rp,rf,bp,bf=flux_at_state(G,pi.copy(),m,args.beta)
                rows.append(dict(m=m,n=2*m,family_index=g['family_index'],graph_seed=g['graph_seed'],r=r,progress_moves=progress,random_proposal_mass=rp,random_accepted_flux=rf,barker_proposal_mass=bp,barker_accepted_flux=bf,random_inverse_flux=(1/rf if rf>0 else np.inf),barker_inverse_flux=(1/bf if bf>0 else np.inf)))
            cert.append({k:g[k] for k in ['m','n','family_index','graph_seed','graph_sha256','delta_m','near_mode_gap']})
            print(f"mechanism m={m}, graph={g['family_index']}",flush=True)
    raw=pd.DataFrame(rows); raw.to_csv(args.output_dir/'barker_flux_raw.csv',index=False)
    summary=raw.groupby(['m','r']).agg(graphs=('family_index','size'),random_proposal_mass=('random_proposal_mass','mean'),barker_proposal_mass=('barker_proposal_mass','mean'),random_accepted_flux=('random_accepted_flux','mean'),barker_accepted_flux=('barker_accepted_flux','mean'),median_random_inverse_flux=('random_inverse_flux','median'),median_barker_inverse_flux=('barker_inverse_flux','median')).reset_index()
    summary.to_csv(args.output_dir/'barker_flux_summary.csv',index=False); pd.DataFrame(cert).to_csv(args.output_dir/'certificates.csv',index=False)
    residual=validate_detailed_balance(firstG,args.beta) if firstG is not None else np.nan
    json.dump({'beta':args.beta,'max_relative_detailed_balance_residual':residual,'note':'Inverse flux is the reciprocal of one-step accepted probability of increasing R at the displayed state; it is not a hitting-time estimate.'},open(args.output_dir/'metadata.json','w'),indent=2)
    print('detailed-balance residual',residual)

def parse():
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,default=ROOT/'results'/'barker_mechanism'); p.add_argument('--m-values',type=int,nargs='+',default=[8,10,12,14]); p.add_argument('--instances-per-size',type=int,default=20); p.add_argument('--bridge-vertex',type=int,default=1); p.add_argument('--beta',type=float,default=2.0); return p.parse_args()
if __name__=='__main__': run(parse())
