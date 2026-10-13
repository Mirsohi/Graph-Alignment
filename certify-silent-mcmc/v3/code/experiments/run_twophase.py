"""Two-phase free-energy estimator on one instance (protein or synthetic).
Writes v2/results/twophase/<case>[_<tag>].json and marginals npz.
Usage: python run_twophase.py --case pscdb_N_74 [--reps 3] [--logT 18] [--k0frac 0.3]
       [--marg_at 1.5,1.52] [--cutoff c8]"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance, save_witness
from gacert2.phase import estimate_two_phase
from gacert.core import log_nfact


def grids(marg_at, bhi, step=0.02, bmaxD=2.6, bminC=1.0):
    gD = np.round(np.arange(0, bmaxD + 1e-9, step), 4)
    gC = np.round(np.arange(bminC, bhi + 1e-9, step), 4)
    extra = np.array(marg_at, float)
    gD = np.unique(np.concatenate((gD, extra[extra <= bmaxD])))
    gC = np.unique(np.concatenate((gC, extra[extra >= bminC])))[::-1]
    return gD, gC


def run(case, reps, logT, k0frac, marg_at, cutoff, bhi, tag, seed0=0):
    A, B, pt, w, meta = load_instance(case, cutoff)
    n = len(w)
    k0 = int(round(k0frac * n))
    gD, gC = grids(marg_at, bhi)
    out = dict(case=case, n=n, k0=k0, k0frac=k0frac, logT=logT, cutoff=cutoff, bhi=bhi, meta=meta,
               log_nfact=log_nfact(n), gridD=gD.tolist(), gridC=gC[::-1].tolist(), reps=[])
    margs = {}
    t0 = time.time()
    refinements = 0
    r = 0
    while r < reps:
        tp, pd, pc = estimate_two_phase(A, B, w, k0, gD, gC, 2 ** logT, 1000 * seed0 + 17 * r + 1,
                                        marg_at=marg_at)
        if tp.best_seen[0] > tp.Sw and refinements < 3:
            # witness refinement: a chain found a better alignment -> adopt it and restart
            w = tp.best_seen[1].copy()
            save_witness(case, w, cutoff)
            refinements += 1
            print(case, 'witness refined', tp.Sw, '->', tp.best_seen[0], flush=True)
            out['reps'], margs, r = [], {}, 0
            meta['Sw'] = int(tp.best_seen[0])
            meta['witness_agree_true'] = float(np.mean(w == pt))
            continue
        rec = dict(FD=tp.FD.tolist(), FC=tp.FC.tolist(), beta_c=tp.beta_c(), ref=tp.ref,
                   fracW_hi=tp.fracW_hi, se_log_fracW=tp.se_log_fracW, best_ref=tp.best_ref, logD=tp.logD, logC=tp.logC,
                   meanS_D=pd['meanS'].tolist(), meanS_C=pc['meanS'][::-1].tolist(),
                   se_D=pd['se'].tolist(), se_C=pc['se'][::-1].tolist(),
                   var_D=pd['varS'].tolist(), var_C=pc['varS'][::-1].tolist(),
                   kmean_D=pd['k_mean'].tolist(), kmean_C=pc['k_mean'][::-1].tolist(),
                   brej_D=pd['brej'].tolist(), brej_C=pc['brej'][::-1].tolist(),
                   best_C=int(pc['best_s']), proposals=tp.proposals, best_D=int(pd['best_s']),
                   wcond_at={str(b): float(tp.wcond(b)) for b in marg_at})
        # boundary occupancy: fraction of retained draws at the region edge
        rec['edge_D'] = [h[k0 - 1] / max(1, sum(h)) for h in pd['k_hist']]
        rec['edge_C'] = [h[k0] / max(1, sum(h)) for h in pc['k_hist']][::-1]
        out['reps'].append(rec)
        for b in marg_at:
            key = round(float(b), 6)
            margs[f'r{r}_D_{b}'] = pd['marg'][key]
            margs[f'r{r}_C_{b}'] = pc['marg'][key]
        out['witness_refinements'] = refinements
        r += 1
        print(case, 'rep', r - 1, 'beta_c', rec['beta_c'], 'fracW', tp.fracW_hi, 'bestC', rec['best_C'],
              'Sw', meta['Sw'], f'{time.time()-t0:.0f}s', flush=True)
    out['seconds'] = time.time() - t0
    out['proposals_per_rep'] = int((len(gD) + len(gC)) * 2 ** logT)
    od = os.path.join(V2, 'results', 'twophase')
    os.makedirs(od, exist_ok=True)
    name = case + (('_' + tag) if tag else '')
    json.dump(out, open(os.path.join(od, name + '.json'), 'w'))
    if margs:
        np.savez_compressed(os.path.join(od, name + '_marg.npz'), **margs)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--logT', type=int, default=18)
    ap.add_argument('--k0frac', type=float, default=0.3)
    ap.add_argument('--marg_at', default='')
    ap.add_argument('--cutoff', default='c8')
    ap.add_argument('--bhi', type=float, default=6.0)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    ma = [float(x) for x in a.marg_at.split(',')] if a.marg_at else []
    for c in a.case.split(','):
        run(c, a.reps, a.logT, a.k0frac, ma, a.cutoff, a.bhi, a.tag)
