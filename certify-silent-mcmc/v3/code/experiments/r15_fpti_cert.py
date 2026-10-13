"""R15 M1: rigorous witness-floor certificates for standard (fixed-path) TI along beta.

FP-TI (the paper's 'standard TI'): unrestricted random-transposition MH (gacert2.phase.ti_path,
kmin=0, kmax=n), started from a seeded uniform random permutation and warm-started along the
grid 0:0.02:beta_max; T proposals per point, 4x on [1, 2.5] (the estimator's schedule);
Lambda_FP(beta) = log n! + trapezoid integral of the per-point mean score.

RIGOROUS floor (deterministic): Lambda(beta) = log sum_pi e^{beta S(pi)} >= beta S* where S* is the
exact integer score of an explicit permutation, S* = max(S(cached witness), best S seen by this
run).  Certified error E_cert(beta) = beta S* - Lambda_FP(beta); E_cert > 0 means the reported
number is certainly below the true Lambda(beta).  Also the transposition-ball floor
beta S* + log sum_{d_H(pi,w*)<=2} e^{beta (S(pi)-S*)} (exact sum), recorded for S1.
TI-BASED context only (proteins): the released two-phase estimate (results/twophase/<case>_all.json).

Usage:
  python r15_fpti_cert.py worker <jobfile> <worker_id>     # claims jobs from a job list
  python r15_fpti_cert.py one <case> <logT> <seed>        # single job (smoke test, tagged)
Job lines: '<case> <logT> <seed>'; case = protein pair name, or 'silent:<tag>'."""
import json, os, sys, time
import numpy as np
from scipy.special import gammaln, logsumexp
from r15_common import RES, V3, prov, dump, claim, append_jsonl
from gacert2.instances import load_instance, synth_pair, find_witness
from gacert2.phase import ti_path, cumtrapz0, log_W_sum
from gacert.core import score

OUT = os.path.join(RES, 'fpti_cert')
TP = os.path.join(V3, 'results', 'twophase')
SG = os.path.join(V3, 'results', 'silent_graph')
EVAL_PROT = [2.0, 2.5, 3.0]
SIL_OFFS = (-0.1, 0.03, 0.08, 0.15)


def instance(case):
    """Returns A, B, w (cached graph-only witness), info dict, grid, eval betas."""
    if case.startswith('silent:'):
        tag = case.split(':', 1)[1]
        rec = json.load(open(os.path.join(SG, tag + '.json')))
        A, B, ps = synth_pair(rec['n'], rec['deg'], rec['q'], rec['seed'])
        t0 = time.time()
        w = find_witness(A, B, restarts=4)          # exp_silent_graph protocol (seed 0, deterministic)
        n = len(w)
        Tw = 2 ** 21
        Sw = int(score(A, B, w))
        if Sw != rec['Sw']:
            raise RuntimeError(f'witness mismatch {tag}: {Sw} vs {rec["Sw"]}')
        bc = rec['beta_c_GT']
        ev = [round(bc + d, 3) for d in SIL_OFFS]
        grid = np.unique(np.concatenate((np.round(np.arange(0, ev[-1] + 1e-9, 0.02), 4), ev)))
        info = dict(kind='silent', tag=tag, beta_c_ref=bc, S_planted=int(score(A, B, ps)),
                    witness_success=bool(Sw >= int(score(A, B, ps))), witness_agree_planted=float(np.mean(w == ps)),
                    witness_search_proposals=int(4 * (Tw + Tw // 8)), witness_search_seconds=time.time() - t0)
        return A, B, w, info, grid, ev
    A, B, pt, w, meta = load_instance(case)
    d = json.load(open(os.path.join(TP, case + '_all.json')))
    info = dict(kind='protein', Sw_cached=int(meta['Sw']), Sw_released=int(d['meta']['Sw']),
                beta_c_2P=[float(r['beta_c']) for r in d['reps']],
                witness_search_proposals=None, witness_search_note='cached graph-only witness (results/witness)')
    grid = np.round(np.arange(0, 3.0 + 1e-9, 0.02), 4)
    return A, B, w, info, grid, EVAL_PROT


def twophase_context(case, betas):
    """TI-BASED (released two-phase estimate) Lambda_2P at betas, per released rep."""
    d = json.load(open(os.path.join(TP, case + '_all.json')))
    bD, bC = np.array(d['gridD']), np.array(d['gridC'])
    out = []
    for r in d['reps']:
        FD, FC = np.array(r['FD']), np.array(r['FC'])
        mD = np.array(r['meanS_D'])
        bCs, FCs = (bC[::-1], FC[::-1]) if bC[0] > bC[-1] else (bC, FC)
        vals = []
        for b in betas:
            fC = np.interp(b, bCs, FCs)
            fD = np.interp(b, bD, FD) if b <= bD[-1] else FD[-1] + (b - bD[-1]) * mD[-1]   # linear extrapolation
            vals.append(float(np.logaddexp(fC, fD)))
        out.append(vals)
    return out


def run(case, logT, seed, tag=''):
    name = case.replace('silent:', 'silent_') + f'_T{logT}_s{seed}' + tag
    path = os.path.join(OUT, name + '.json')
    if os.path.exists(path):
        return None
    t0 = time.time()
    A, B, w, info, grid, ev = instance(case)
    n = len(w)
    Sw = int(score(A, B, w))
    t1 = time.time()
    out = ti_path(A, B, w, grid, np.random.default_rng(seed).permutation(n), 0, n, 2 ** logT, seed)
    secs_ti = time.time() - t1
    Lfp = gammaln(n + 1) + cumtrapz0(out['beta'], out['meanS'])
    best_s = int(out['best_s'])
    assert int(score(A, B, out['best_p'])) == best_s
    if best_s > Sw:
        wstar, Sstar, src = out['best_p'], best_s, 'fpti_best'
    else:
        wstar, Sstar, src = w, Sw, 'cached_witness'
    cert = grid * Sstar - Lfp
    ball = np.array([b * Sstar + log_W_sum(A, B, wstar, float(b)) for b in grid])
    cert_ball = ball - Lfp
    knuc = out['k_mean'] > 0.3 * n
    first = lambda c: float(grid[np.argmax(c > 0)]) if (c > 0).any() else None
    idx = [int(np.argmin(np.abs(grid - b))) for b in ev]
    rec = dict(case=case, name=name, n=n, logT_per_point=logT, seed=seed, info=info,
               config=dict(grid=[float(grid[0]), float(grid[-1]), 0.02], boost='4x on [1, 2.5]', start='uniform random perm, default_rng(seed)',
                           kernel='random-transposition MH, unrestricted (ti_path kmin=0,kmax=n)', burn_frac=0.25, anchor='log n!',
                           quadrature='trapezoid'),
               proposals_ti=int(out['T']), log2_proposals_ti=float(np.log2(out['T'])), seconds_ti=secs_ti,
               Sw_cached=Sw, S_star=Sstar, S_star_source=src,
               path=dict(beta=out['beta'], meanS=out['meanS'], se=out['se'], k_mean=out['k_mean'], smax=out['smax'],
                         acc=out['acc'], Lambda_FP=Lfp),
               RIGOROUS=dict(note='deterministic floors; E_cert>0 => Lambda_FP(beta) is certainly below Lambda(beta)',
                             E_cert_point=cert, E_cert_ball=cert_ball,
                             first_cert_beta_point=first(cert), first_cert_beta_ball=first(cert_ball),
                             eval=[dict(beta=float(grid[i]), E_cert_point=float(cert[i]), E_cert_ball=float(cert_ball[i]),
                                        Lambda_FP=float(Lfp[i]), floor_point=float(grid[i] * Sstar), floor_ball=float(ball[i]))
                                   for i in idx]),
               nucleation_beta=float(grid[np.argmax(knuc)]) if knuc.any() else None)
    if info['kind'] == 'protein':
        ctx = twophase_context(case, [float(grid[i]) for i in idx])
        rec['TI_BASED_context'] = dict(note='released two-phase estimate Lambda_2P (TI-based, per released rep); '
                                            'est_err = Lambda_2P - Lambda_FP. Context only.',
                                       Lambda_2P=ctx, est_err=[[c - float(Lfp[i]) for c, i in zip(cr, idx)] for cr in ctx])
    rec['prov'] = prov(__file__, t0)
    dump(rec, path)
    summ = dict(name=name, n=n, S_star=Sstar, src=src, logprop=rec['log2_proposals_ti'], secs=rec['prov']['wall_seconds'],
                first=rec['RIGOROUS']['first_cert_beta_point'], nuc=rec['nucleation_beta'],
                ecert=[round(e['E_cert_point'], 2) for e in rec['RIGOROUS']['eval']])
    print(json.dumps(summ), flush=True)
    return rec


if __name__ == '__main__':
    if sys.argv[1] == 'one':
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), tag=(sys.argv[5] if len(sys.argv) > 5 else ''))
    else:
        jobfile, wid = sys.argv[2], sys.argv[3]
        jobs = [l.split() for l in open(jobfile) if l.strip() and not l.startswith('#')]
        lock = os.path.join(OUT, 'locks')
        for case, logT, seed in jobs:
            nm = case.replace('silent:', 'silent_') + f'_T{logT}_s{seed}'
            if os.path.exists(os.path.join(OUT, nm + '.json')) or not claim(lock, nm):
                continue
            t = time.time()
            try:
                run(case, int(logT), int(seed))
                append_jsonl(dict(job=nm, worker=wid, secs=time.time() - t, ok=True), os.path.join(OUT, 'jobs_done.jsonl'))
            except Exception as e:
                import traceback; traceback.print_exc()
                append_jsonl(dict(job=nm, worker=wid, ok=False, err=repr(e)), os.path.join(OUT, 'jobs_done.jsonl'))
