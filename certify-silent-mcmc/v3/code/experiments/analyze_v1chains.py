"""Certificates and calibrated statements on the re-run v1 protocol chains
(results/v1chains/*.json), using only the graph-only witness:
  rigorous certificate: lambda = 0 bound < 0.01 (per chain, statistic = max retained S);
  calibrated tilted certificate: v1 TI Lambda-hat (lambda <= min(beta, 1.2)) with the
      witness floor Lambda(l) >= l S(w) - log n!;
  estimated TV lower bound (calibrated): max(nu(D) - mu^(D), nu(C) - mu^(C)) using the
      two-phase weights, reported ONLY for pairs that pass the validity check.
Writes results/summary_v1chains.json.  Usage: python analyze_v1chains.py"""
import json, os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(V2)
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.phase import TwoPhase
from gacert.core import split_rhat, log_nfact

LN01 = np.log(0.01)
R = os.path.join(V2, 'results')
valid = {}
for f in glob.glob(os.path.join(R, 'validity', '*_s1.json')):
    v = json.load(open(f))
    valid[v['case']] = v
v22 = {}
for f in glob.glob(os.path.join(R, 'validity22', '*_s2.json')):
    v = json.load(open(f))
    v22[v['case']] = v
vc = json.load(open(os.path.join(R, 'validity_c_all.json')))
status = {}
for c, v in valid.items():
    ch = v['checks']
    mid = any(b['mass'] > 0.05 and 0.1 * v['n'] < b['k_peak'] < v['k0'] + 0.05 * v['n'] for b in v['basins'])
    if not ch['edge_ok'] or not ch['melt_ok'] or mid:
        status[c] = 'multi-phase'
    elif c in vc and not vc[c]['ok']:
        status[c] = 'C-metastable'
    elif ch['profile_ok']:
        status[c] = 'pass'
    else:
        # second window pass at 2^22: accept if it agrees with the estimator (|dlogit| < 1)
        if c in v22 and v22[c]['checks']['profile_ok']:
            status[c] = 'pass (2nd window pass)'
        elif c in v22:
            status[c] = 'window-inconclusive'
        else:
            status[c] = 'window-inconclusive (pending)'
out = dict(status=status, pairs={}, table={})
for f in sorted(glob.glob(os.path.join(R, 'v1chains', '*.json'))):
    d = json.load(open(f))
    case, n, k0, Sw = d['case'], d['n'], d['k0'], d['Sw']
    v1 = json.load(open(os.path.join(ROOT, 'results', 'protein', case + '.json')))
    lams, L = np.array(v1['ti']['lams']), np.array(v1['ti']['Lambda'])
    lnf = log_nfact(n)
    L = np.maximum(L, lams * Sw - lnf)
    tp = None
    st = status.get(case, 'unknown')
    if st.startswith('pass'):
        e = json.load(open(os.path.join(R, 'twophase', f'{case}_all.json')))
        tps = [TwoPhase(n, 0, 0, 0, e['gridD'], r['FD'], e['gridC'], r['FC']) for r in e['reps']]
    rows = []
    for g in d['groups']:
        b = g['beta']
        Ls = min(len(c['S']) for c in g['chains'])
        rh = split_rhat(np.array([c['S'][:Ls] for c in g['chains']], float))
        wC = None
        if st.startswith('pass') and b <= e['gridD'][-1]:
            wC = float(np.mean([t.wcond(b) for t in tps]))
        for c in g['chains']:
            crude = lnf + b * (c['maxS'] - Sw)
            ok = lams <= b
            tilt = float(np.min(lnf + L[ok] + (b - lams[ok]) * c['maxS'] - b * Sw))
            tv = None
            if wC is not None:
                tv = max(c['nuD'] - (1 - wC), (1 - c['nuD']) - wC)
            rows.append(dict(beta=b, start=g['start'], nuD=c['nuD'], rhat=rh, crude_cert=bool(crude < LN01),
                             tilted_cert=bool(tilt < LN01), tv_est=tv, wC=wC, rhat_pass=bool(rh < 1.01)))
    out['pairs'][case] = dict(status=st, rows=rows)
    for r in rows:
        key = f"{r['start']}_{r['beta']}"
        t = out['table'].setdefault(key, dict(chains=0, in_D=0, crude=0, tilted=0, tv_eval=0, tv_ge_05=0,
                                              rhat_pass_in_wrong=0, wrong_eval=0))
        t['chains'] += 1
        t['in_D'] += int(r['nuD'] > 0.5)
        t['crude'] += int(r['crude_cert'])
        t['tilted'] += int(r['tilted_cert'])
        if r['tv_est'] is not None:
            t['tv_eval'] += 1
            t['tv_ge_05'] += int(r['tv_est'] >= 0.5)
            if r['tv_est'] >= 0.5:
                t['wrong_eval'] += 1
                t['rhat_pass_in_wrong'] += int(r['rhat_pass'])
json.dump(out, open(os.path.join(R, 'summary_v1chains.json'), 'w'), indent=1)
print('status:', json.dumps(status, indent=0))
for k in sorted(out['table'], key=lambda s: (s.split('_')[0], float(s.split('_')[1]))):
    print(k, out['table'][k])
for c in ('pscdb_N_141', 'pscdb_N_261', 'pscdb_N_75', 'pscdb_N_139', 'pscdb_N_217', 'pscdb_N_271'):
    if c in out['pairs']:
        for r in out['pairs'][c]['rows']:
            if r['beta'] in (1.5, 1.75) and r['tv_est'] is not None and r['tv_est'] >= 0.5:
                print(c, r)
