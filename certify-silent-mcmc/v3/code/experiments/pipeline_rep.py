"""End-to-end replicate of the two-phase pipeline with fresh seeds for EVERY stage:
graph-only witness search (4 annealing restarts), restricted TI along both paths,
witness refinement, nested multilevel reference at beta_hi.  Records the estimate at
the ground-truth beta values plus a per-component error budget:
  ref  : SE of log mu(W|C) from the nested reference (batch means per level);
  mc   : Monte Carlo SE of F_D(b), F_C(b) from per-grid batch SEs, trapezoid weights,
         grid points treated as independent;
  disc : half-width of the left/right Riemann bracket computed with the ESTIMATED
         restricted means (Prop. 2(ii) is rigorous only for exact means);
propagated to logit w_C (= F_C - F_D), log Z(beta) (delta method, weights w_C, 1-w_C)
and beta_c (divided by the latent heat |E_C S - E_D S| at beta_c).
Also records proposal counts for witness search, estimator and nested reference.
Usage: python pipeline_rep.py --case pscdb_O_19 --seed 1"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
sys.path.insert(0, HERE)
from gacert2.instances import load_instance, find_witness
from gacert2.phase import estimate_two_phase, nested_reference, log_W_sum, TwoPhase
from gacert.core import score
from run_twophase import grids

OUT = os.path.join(V2, 'results', 'pipeline_reps')
os.makedirs(OUT, exist_ok=True)


def mc_var_path(b, se, lo, hi):
    """variance of int_lo^hi f by trapezoid with independent per-point SEs"""
    b, se = np.asarray(b), np.asarray(se)
    sel = (b >= lo - 1e-9) & (b <= hi + 1e-9)
    bb, ss = b[sel], se[sel]
    if len(bb) < 2:
        return 0.0
    h = np.diff(bb)
    wts = np.zeros(len(bb))
    wts[:-1] += h / 2
    wts[1:] += h / 2
    return float(np.sum((wts * ss) ** 2))


def disc_half(b, m, lo, hi):
    b, m = np.asarray(b), np.asarray(m)
    sel = (b >= lo - 1e-9) & (b <= hi + 1e-9)
    bb, mm = b[sel], m[sel]
    return float(0.5 * np.sum(np.diff(bb) * np.abs(np.diff(mm))))


def run(case, seed, logT=18, k0frac=0.3, bhi=6.0, rules='transition'):
    path = os.path.join(OUT, f'{case}_s{seed}.json')
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    A, B, pt, _, meta = load_instance(case)
    n = A.shape[0]
    k0 = int(round(k0frac * n))
    gt = json.load(open(os.path.join(V2, 'results', 'gt_comparison.json')))[case]
    betas = [r['beta'] for r in gt['rows']]
    Tw = 2 ** 21 if n <= 110 else 2 ** 23
    w = find_witness(A, B, restarts=4, T=Tw, seed=1000 + seed)
    cost = dict(witness=int(4 * (Tw + Tw // 8)), estimator=0, nested=0, refinement_restarts=0)
    gD, gC = grids(betas, bhi)
    for attempt in range(3):
        tp, pd, pc = estimate_two_phase(A, B, w, k0, gD, gC, 2 ** logT, 50000 + 97 * seed + attempt)
        cost['estimator'] += int(tp.proposals)
        if tp.best_seen[0] > tp.Sw:
            w = tp.best_seen[1].copy()
            cost['refinement_restarts'] += 1
            continue
        break
    Sw = int(score(A, B, w))
    Tn = 2 ** 22
    lm, lev, fr, se_ref, bs, bp, fb = nested_reference(A, B, w, k0, bhi, Tn, 70000 + seed)
    cost['nested'] = int((len(lev) - 1) * (Tn // 4 + Tn))
    ref = bhi * Sw + log_W_sum(A, B, w, bhi) - lm
    FC = np.array(tp.FC) - tp.ref + ref
    tp = TwoPhase(n, Sw, tp.logD, tp.logC, tp.bD, tp.FD, tp.bC, FC)
    bc = tp.beta_c()
    bD, mD, sD = pd['beta'], pd['meanS'], pd['se']
    bC, mC, sC = pc['beta'][::-1], pc['meanS'][::-1], pc['se'][::-1]
    rows = []
    for b in betas + ([bc] if np.isfinite(bc) else []):
        vD = mc_var_path(bD, sD, 0.0, b)
        vC = mc_var_path(bC, sC, b, bhi)
        dD = disc_half(bD, mD, 0.0, b)
        dC = disc_half(bC, mC, b, bhi)
        wc = float(tp.wcond(b))
        lz = float(np.logaddexp(tp.fC(b), tp.fD(b)))
        rows.append(dict(beta=float(b), wC=wc, logit=float(tp.fC(b) - tp.fD(b)), logZ=lz,
                         se_logit_mc=float(np.sqrt(vD + vC)), se_logit_ref=float(se_ref),
                         disc_logit_half=dD + dC,
                         se_logZ_mc=float(np.sqrt(wc ** 2 * vC + (1 - wc) ** 2 * vD)), se_logZ_ref=float(wc * se_ref),
                         disc_logZ_half=float(wc * dC + (1 - wc) * dD)))
    lat = float(abs(np.interp(bc, bC, mC) - np.interp(bc, bD, mD))) if np.isfinite(bc) else float('nan')
    rec = dict(case=case, seed=seed, n=n, k0=k0, Sw=Sw, Sw_cached=meta['Sw'], witness_agree_true=float(np.mean(w == pt)),
               beta_c=bc, latent_heat=lat, nested=dict(log_mu_W=lm, levels=lev, se=se_ref, floor_bound=fb, best=bs),
               rows=rows, cost=cost, cost_total=int(sum(v for k, v in cost.items() if k != 'refinement_restarts')),
               sec=time.time() - t0)
    if rules != 'none' and np.isfinite(bc):
        # released default: frozen rules at beta_c, two seeds (gacert2.diagnose); edge occupancy
        # from the estimator's own restricted chains at the grid point nearest beta_c.
        from gacert2.diagnose import diagnose
        iD = int(np.argmin(np.abs(np.array(pd['beta']) - bc)))
        iC = int(np.argmin(np.abs(np.array(pc['beta']) - bc)))
        hD, hC = pd['k_hist'][iD], pc['k_hist'][iC]
        edges = (hD[k0 - 1] / max(1, sum(hD)), hC[k0] / max(1, sum(hC)))
        dg = diagnose(A, B, w, k0, bc, 0.5, seeds=(100 + seed, 200 + seed), edges=edges)
        rec['diagnostic'] = dict(rules=rules, status=dg['status'],
                                 failed_rules=[[k for k, v in r['rules'].items() if not v] for r in dg['runs']],
                                 proposals=int(sum(r['proposals'] for r in dg['runs'])))
        if rules == 'extended':
            rec['diagnostic']['note'] = 'extended off-transition checks: see validity_multi.py'
    json.dump(rec, open(path, 'w'), default=float)
    r_bc = rows[-1] if np.isfinite(bc) else None
    print(case, seed, f'Sw={Sw} (cached {meta["Sw"]}) bc={bc:.4f}', 'wC', [round(r['wC'], 3) for r in rows[:len(betas)]],
          (f"bc-row: se_mc {r_bc['se_logit_mc']:.3f} se_ref {r_bc['se_logit_ref']:.3f} disc {r_bc['disc_logit_half']:.3f}"
           if r_bc else ''), f'log2 cost {np.log2(rec["cost_total"]):.2f}', f'{rec["sec"]:.0f}s', flush=True)
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--seeds', default='1,2,3')
    ap.add_argument('--rules', default='transition', choices=('transition', 'extended', 'none'),
                    help='validity diagnostic after the estimate (default: frozen rules at beta_c, two seeds)')
    a = ap.parse_args()
    for c in a.case.split(','):
        for s in a.seeds.split(','):
            run(c, int(s), rules=a.rules)
