"""Replace the single-stage reference log mu_{bhi|C}(W) of saved two-phase runs by
the nested multilevel estimate (phase.nested_reference) and shift F_C exactly
(F_C = ref - integral; the integral is unchanged).  One nested estimate per replicate.
Writes <case>_<tag>.json in place with fields ref_single, ref_nested, beta_c_single.
Usage: python fix_reference.py tag case1,case2,...|all"""
import json, os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.phase import nested_reference, log_W_sum, TwoPhase
from gacert.core import score

tag = sys.argv[1]
files = sorted(glob.glob(os.path.join(V2, 'results', 'twophase', f'*_{tag}.json')))
if len(sys.argv) > 2 and sys.argv[2] != 'all':
    want = set(sys.argv[2].split(','))
    files = [f for f in files if os.path.basename(f)[:-len(tag) - 6] in want]
for f in files:
    d = json.load(open(f))
    if d.get('reference') == 'nested':
        continue
    A, B, pt, w, meta = load_instance(d['case'], d.get('cutoff', 'c8'))
    Sw = int(score(A, B, w))
    assert Sw == d['meta']['Sw'], (d['case'], Sw, d['meta']['Sw'])
    n, k0, bhi = d['n'], d['k0'], d['bhi']
    lw = log_W_sum(A, B, w, bhi)
    for i, r in enumerate(d['reps']):
        lm, lev, fr, se, bs, bp, fb = nested_reference(A, B, w, k0, bhi, 2 ** 22, 777 + 13 * i)
        ref_new = bhi * Sw + lw - lm
        r['ref_single'] = r['ref']
        r['beta_c_single'] = r['beta_c']
        r['FC'] = (np.array(r['FC']) - r['ref'] + ref_new).tolist()
        r['ref'] = ref_new
        r['ref_nested'] = dict(log_mu_W=lm, levels=lev, fracs=fr, se=se, best=bs, floor_bound=fb)
        tp = TwoPhase(n, Sw, r['logD'], r['logC'], d['gridD'], r['FD'], d['gridC'], r['FC'])
        r['beta_c'] = tp.beta_c()
        for b in list(r.get('wcond_at', {})):
            r['wcond_at'][b] = float(tp.wcond(float(b)))
        if bs > Sw:
            print('  WARNING better state found in nested reference', bs, '>', Sw)
        print(d['case'], i, f"ref shift {ref_new - r['ref_single']:+.2f} nats (se {se:.2f}); "
              f"beta_c {r['beta_c_single']:.4f} -> {r['beta_c']:.4f}; levels {lev}", flush=True)
    d['reference'] = 'nested'
    json.dump(d, open(f, 'w'))
