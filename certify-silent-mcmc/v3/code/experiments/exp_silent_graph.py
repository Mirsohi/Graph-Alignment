"""GT-labelled SILENT-regime diagnostic table on synthetic correlated graphs.

Why a new regime: at the model-consistent beta(q) (exp_alarm.py) the diffuse phase
of sparse ER pairs is glassy, so random-start chains disagree and R-hat catches
them.  Here beta is a tempering hyper-parameter (as for the protein pairs) and we
evaluate just ABOVE the ground-truth transition beta_c^GT, where the diffuse phase
is a liquid: random-start chains mix inside it, agree with each other and pass
R-hat, yet the posterior mass is in the condensed phase.  One beta below beta_c
supplies correct random-start chains (false-alarm population).

Ground truth (label side only; may use the planted pi*): stratified overlap-window
profile around pi* (4 seeds x 2^20 per window) gives w*_GT(beta) = posterior mass of
{overlap with pi* >= 0.3 n}; beta_c^GT from a coarse window scan.  A chain is WRONG
if |nu*_chain - w*_GT| > 0.5 and CORRECT if < 0.2 (else excluded), with nu*_chain the
fraction of recorded draws with overlap with pi* >= 0.3 n.  A group (4 chains) is
WRONG if all its labelled chains are wrong, CORRECT if all are correct.
Diagnostics (graph-only; pi* is never used): group split-R-hat(S) >= 1.01, R* >= 1.05,
D/D_iid >= 1.5; chain: rigorous lambda=0 certificate, TI-based tilted bound (v1 TI
with witness floor), two-phase TV bound |nu_chain(C) - w-hat_C| >= 0.5.
Usage: python exp_silent_graph.py [i0 i1]"""
import json, os, sys, time
import numpy as np
from scipy.special import logsumexp
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
sys.path.insert(0, HERE)
from gacert2.instances import synth_pair, find_witness
from gacert2.windows import overlap_profile
from gacert2.phase import rmh, estimate_two_phase, nested_reference, log_W_sum
from gacert.core import score, split_rhat, d_marg, iid_floor_plugin, log_mgf_ti, log_nfact
from exp_alarm import rstar

LN01 = np.log(0.01)
OUT = os.path.join(V2, 'results', os.environ.get('SILENT_OUT', 'silent_graph'))
os.makedirs(OUT, exist_ok=True)


def w_gt(A, B, ps, beta, k0, T, seed):
    lp = overlap_profile(A, B, ps, beta, T, seed)['logp']
    f = np.isfinite(lp[k0:])
    return float(np.exp(logsumexp(lp[k0:][f])))


def run_instance(n, deg, q, seed, T=2 ** 20, K=4):
    tag = f'n{n}_d{deg}_q{q}_s{seed}'
    path = os.path.join(OUT, tag + '.json')
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    A, B, ps = synth_pair(n, deg, q, seed)
    k0 = int(round(0.3 * n))
    w = find_witness(A, B, restarts=4)   # default protocol budget (2^21 steps per restart for n <= 110)
    Sw = int(score(A, B, w))
    # coarse GT scan for beta_c^GT (label side)
    grid = np.round(np.arange(0.6, 3.01, 0.05), 3)
    wg = np.array([w_gt(A, B, ps, b, k0, 2 ** 17, 5) for b in grid])
    lg = np.log(np.clip(wg, 1e-9, 1) / np.clip(1 - wg, 1e-9, 1))
    i = np.where(np.diff(np.sign(lg)) > 0)[0]
    if not len(i):
        rec = dict(tag=tag, note='no GT crossing in [0.6, 3]', Sw=Sw, S_planted=int(score(A, B, ps)))
        json.dump(rec, open(path, 'w'))
        print(tag, rec, flush=True)
        return rec
    i = i[-1]
    bcg = float(grid[i] - lg[i] * (grid[i + 1] - grid[i]) / (lg[i + 1] - lg[i]))
    betas = [round(bcg + d, 3) for d in (-0.1, 0.03, 0.08, 0.15)]
    # two-phase estimator (one run, nested reference), graph-only
    gD = np.round(np.arange(0, max(betas) + 0.021, 0.02), 4)
    gC = np.round(np.arange(max(0.6, min(betas) - 0.2), 6.0001, 0.02), 4)[::-1]
    tp, pd, pc = estimate_two_phase(A, B, w, k0, gD, gC, 2 ** 15, seed + 7)
    lm, *_ = nested_reference(A, B, w, k0, float(gC[0]), 2 ** 20, seed + 9)
    tp.FC = tp.FC - tp.ref + (float(gC[0]) * Sw + log_W_sum(A, B, w, float(gC[0])) - lm)
    # v1 TI (lambda <= beta) with witness floor, for the tilted bound
    lams = np.round(np.arange(0, max(betas) + 0.0001, 0.05), 4)
    Lam, ms, sd = log_mgf_ti(A, B, lams, T=2 ** 18, seeds=(11, 12), rng_seed=seed)
    lnf = log_nfact(n)
    Lam = np.maximum(Lam, lams * Sw - lnf)
    rng = np.random.default_rng(seed + 99)
    rows = []
    for beta in betas:
        ws = [w_gt(A, B, ps, beta, k0, 2 ** 20, s) for s in (1, 2, 3, 4)]
        wG = float(np.mean(ws))
        wE = float(tp.wcond(beta))
        ok = lams <= beta + 1e-9
        groups = []
        for start in ('random', 'witness'):
            ch, Ms, feats, Ss = [], [], [], []
            for c in range(K):
                p0 = rng.permutation(n) if start == 'random' else w.copy()
                st, kt, M, *_ = rmh(A, B, p0, w, beta, T, T // 4, (T - T // 4) // 1024, 0, n,
                                    int(rng.integers(2 ** 31)), True)
                nuC = float(np.mean(kt >= k0))
                # label side: overlap with pi* per recorded draw is not stored; use the
                # time-averaged agreement with pi* (phases are well separated in overlap)
                ov = float(M[np.arange(n), ps].mean())
                nu_star = float(ov >= 0.3) if (ov < 0.2 or ov > 0.5) else np.nan
                Mx = float(st.max())
                crude = lnf + beta * (Mx - Sw)
                tilt = float(np.min(lnf + Lam[ok] + (beta - lams[ok]) * Mx - beta * Sw))
                tv = abs(nuC - wE)
                if np.isnan(nu_star):
                    lab = 'ambiguous'
                else:
                    dv = abs(nu_star - wG)
                    lab = 'wrong' if dv > 0.5 else ('correct' if dv < 0.2 else 'ambiguous')
                ch.append(dict(label=lab, nuC=nuC, ov_star=ov, maxS=Mx, crude_cert=bool(crude < LN01),
                               tilted_cert=bool(tilt < LN01), twophase_tv_alarm=bool(tv >= 0.5), tv_est=tv))
                Ms.append(M)
                Ss.append(st)
                feats.append(np.column_stack((st, kt)))
            L = min(len(s) for s in Ss)
            P = np.mean(Ms, 0)
            rh = split_rhat(np.array([s[:L] for s in Ss]))
            rs = rstar([f[:L] for f in feats])
            dr = d_marg(Ms) / max(iid_floor_plugin(P, L), 1e-12)
            labs = [c['label'] for c in ch if c['label'] != 'ambiguous']
            glab = ('wrong' if labs and all(l == 'wrong' for l in labs) else
                    'correct' if labs and all(l == 'correct' for l in labs) else 'mixed')
            groups.append(dict(start=start, label=glab, chains=ch, rhat=float(rh), rstar=rs, d_ratio=float(dr),
                               rhat_alarm=bool(not rh < 1.01), rstar_alarm=bool(rs >= 1.05), d_alarm=bool(dr >= 1.5)))
        rows.append(dict(beta=beta, dbeta=round(beta - bcg, 3), w_GT=wG, w_GT_seeds=ws,
                         w_GT_se=float(np.std(ws, ddof=1) / 2), w_twophase=wE, groups=groups))
        print(tag, f'b={beta:.3f} (bc+{beta - bcg:+.2f}) wGT={wG:.3f}±{np.std(ws, ddof=1) / 2:.3f} w2P={wE:.3f}',
              [(g['start'][0], g['label'][0], ''.join(c['label'][0] for c in g['chains']), round(g['rhat'], 3),
                round(g['rstar'], 2), round(g['d_ratio'], 2), ''.join(str(int(c['crude_cert'])) for c in g['chains']),
                ''.join(str(int(c['tilted_cert'])) for c in g['chains']),
                ''.join(str(int(c['twophase_tv_alarm'])) for c in g['chains'])) for g in groups], flush=True)
    rec = dict(tag=tag, n=n, deg=deg, q=q, seed=seed, k0=k0, Sw=Sw, S_planted=int(score(A, B, ps)),
               witness_agree_planted=float(np.mean(w == ps)), beta_c_GT=bcg, beta_c_2P=float(tp.beta_c()),
               gt_scan=dict(beta=grid.tolist(), w=wg.tolist()), rows=rows, sec=time.time() - t0)
    json.dump(rec, open(path, 'w'))
    print(tag, f'done bcGT={bcg:.3f} bc2P={rec["beta_c_2P"]:.3f} {rec["sec"]:.0f}s', flush=True)
    return rec


CFGS = [(n, d, q, s) for (d, q) in ((7, 0.03), (8, 0.08)) for n in (40, 60) for s in (0, 1, 2)]
CFGS += [(n, d, q, 3) for (d, q) in ((7, 0.03), (8, 0.08)) for n in (40, 60)]

if __name__ == '__main__':
    i0, i1 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, len(CFGS))
    for c in CFGS[i0:i1]:
        run_instance(*c)
