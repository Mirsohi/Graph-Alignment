"""R15 M2 pipeline arm: the frozen released procedure (logic of pipeline_rep.run) on array instances.
Per replicate rep (seeds 1, 2): graph-only witness search (4 restarts, Tw = 2^21, seed 1000+rep);
two-phase estimator (k0 = round(0.3 n), b_hi = 6, step 0.02, 2^18 per point, 4x on [1, 2.5], grids as
run_twophase.grids with marg_at = G), up to 3 attempts with witness refinement (seed 50000+97 rep+attempt);
nested reference at b_hi (Tn = 2^22, seed 70000+rep); released screen gacert2.diagnose.diagnose_bounded at
beta_c (seeds (100+rep, 200+rep), one retry with (300+rep, 400+rep)), edge occupancy from the estimator's
chains, w_hat = 0.5 (as pipeline_rep).
Estimate on G: a_hat(beta) = w_C(beta) a_C + (1 - w_C(beta)) a_D, a_R = (1/n) sum_i M_R[i, pi*(i)]
(pi* used ONLY here, for evaluation output; nothing in the procedure reads it).
Writes results/r15/p1_heldout/pipe/<inst>_r<rep>.json (+ _marg.npz with float32 restricted marginals).
Usage: python r15_p1_pipeline.py <inst> <rep> [outdir]"""
import os, sys, time
import numpy as np
from r15_common import prov, dump
from r15_p1_common import P1, G, make, a_from_marg
from gacert2.instances import find_witness
from gacert2.phase import estimate_two_phase, nested_reference, log_W_sum, TwoPhase
from gacert2.diagnose import diagnose_bounded
from gacert.core import score
from run_twophase import grids


def run(inst, rep, outdir=None, logT=18, k0frac=0.3, bhi=6.0):
    outdir = outdir or os.path.join(P1, 'pipe')
    path = os.path.join(outdir, f'{inst}_r{rep}.json')
    if os.path.exists(path):
        return None
    t0 = time.time()
    A, B, ps = make(inst)
    n = A.shape[0]
    k0 = int(round(k0frac * n))
    Tw = 2 ** 21 if n <= 110 else 2 ** 23
    tw = time.time()
    w = find_witness(A, B, restarts=4, T=Tw, seed=1000 + rep)
    secs = dict(witness=time.time() - tw)
    cost = dict(witness=int(4 * (Tw + Tw // 8)), estimator=0, refinements=0, nested=0, diagnostics=0, retry=0)
    attempt_cost = []
    refinements = 0
    gD, gC = grids(G, bhi)
    te = time.time()
    for attempt in range(3):
        tp, pd, pc = estimate_two_phase(A, B, w, k0, gD, gC, 2 ** logT, 50000 + 97 * rep + attempt, marg_at=G)
        attempt_cost.append(int(tp.proposals))
        if tp.best_seen[0] > tp.Sw:
            w = tp.best_seen[1].copy()
            refinements += 1
            continue
        break
    secs['estimator'] = time.time() - te
    cost['estimator'] = attempt_cost[-1]                 # final (used) estimator attempt
    cost['refinements'] = int(sum(attempt_cost[:-1]))     # earlier attempts abandoned after a witness refinement
    Sw = int(score(A, B, w))
    Tn = 2 ** 22
    tn = time.time()
    lm, lev, fr, se_ref, bs, bp, fb = nested_reference(A, B, w, k0, bhi, Tn, 70000 + rep)
    secs['nested'] = time.time() - tn
    cost['nested'] = int((len(lev) - 1) * (Tn // 4 + Tn))
    ref = bhi * Sw + log_W_sum(A, B, w, bhi) - lm
    FC = np.array(tp.FC) - tp.ref + ref
    tp = TwoPhase(n, Sw, tp.logD, tp.logC, tp.bD, tp.FD, tp.bC, FC)
    bc = tp.beta_c()
    # estimate on G (marginals of the restricted chains of the final attempt)
    est, margs = [], {}
    for b in G:
        key = round(float(b), 6)
        MD, MC = pd['marg'][key], pc['marg'][key]
        wc = float(tp.wcond(b))
        aD, aC = a_from_marg(MD, ps), a_from_marg(MC, ps)
        est.append(dict(beta=b, wC=wc, logit=float(tp.fC(b) - tp.fD(b)), a_D=aD, a_C=aC, a_hat=wc * aC + (1 - wc) * aD))
        margs[f'{b}_D'] = MD.astype(np.float32)
        margs[f'{b}_C'] = MC.astype(np.float32)
    rec = dict(inst=inst, rep=rep, n=n, k0=k0, Sw=Sw, S_planted=int(score(A, B, ps)),
               witness_agree_planted=float(np.mean(w == ps)), refinements=refinements, witness=w.tolist(),
               beta_c=bc, crossing=bool(np.isfinite(bc)), estimate=est,
               nested=dict(log_mu_W=lm, levels=lev, se=se_ref, floor_bound=fb),
               config=dict(logT=logT, k0frac=k0frac, bhi=bhi, Tw=Tw, Tn=Tn, G=G, seeds=dict(
                   witness=1000 + rep, estimator=[50000 + 97 * rep + a for a in range(3)], nested=70000 + rep,
                   diag=(100 + rep, 200 + rep), retry=(300 + rep, 400 + rep))))
    if np.isfinite(bc):
        iD = int(np.argmin(np.abs(np.array(pd['beta']) - bc)))
        iC = int(np.argmin(np.abs(np.array(pc['beta']) - bc)))
        hD, hC = pd['k_hist'][iD], pc['k_hist'][iC]
        edges = (hD[k0 - 1] / max(1, sum(hD)), hC[k0] / max(1, sum(hC)))
        td = time.time()
        dg = diagnose_bounded(A, B, w, k0, bc, 0.5, seeds=(100 + rep, 200 + rep), retry_seeds=(300 + rep, 400 + rep),
                              edges=edges)
        secs['diagnostics_total'] = time.time() - td
        atts = []
        for j, at in enumerate(dg['attempts']):
            runs = []
            for r in at['runs']:
                runs.append(dict(rules=r['rules'], passed=r['passed'], edge_D=r['edge_D'], edge_C=r['edge_C'],
                                 w_window=r['w_window'], dS_D=r['dS_D'], se_D=r['se_D'], dS_C=r['dS_C'], se_C=r['se_C'],
                                 basins=r['basins'], proposals=r['proposals']))
            atts.append(dict(status=at['status'], runs=runs))
            cost['diagnostics' if j == 0 else 'retry'] += int(sum(r['proposals'] for r in at['runs']))
        rec['screen'] = dict(status=dg['status'], retried=dg['retried'], attempts=atts, w_hat_at_bc=float(tp.wcond(bc)),
                             w_hat_used=0.5, edges=edges)
    else:
        rec['screen'] = dict(status='no-crossing', retried=False, attempts=[])
    rec['cost'] = cost
    rec['cost_total'] = int(sum(cost.values()))
    rec['seconds'] = secs
    rec['prov'] = prov(__file__, t0)
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, f'{inst}_r{rep}_marg.npz'), **margs)
    dump(rec, path)
    print(inst, rep, f'bc={bc:.3f} Sw={Sw}/{rec["S_planted"]} status={rec["screen"]["status"]} '
          f'a_hat={[round(e["a_hat"], 3) for e in est]} log2cost={np.log2(rec["cost_total"]):.2f} '
          f'{rec["prov"]["wall_seconds"]:.0f}s', flush=True)
    return rec


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]), outdir=(sys.argv[3] if len(sys.argv) > 3 else None),
        logT=(int(sys.argv[4]) if len(sys.argv) > 4 else 18))
