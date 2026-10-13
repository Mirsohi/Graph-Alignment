"""Observed phase-switch counts of plain unrestricted MH (random and witness
start) at given betas; hysteresis thresholds k <= 0.15 n / k >= 0.5 n.
Usage: python switch_counts.py --case pscdb_O_19 --betas 1.44,1.46 --logT 28"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.phase import switch_count

ap = argparse.ArgumentParser()
ap.add_argument('--case'); ap.add_argument('--betas'); ap.add_argument('--logT', type=int, default=28)
a = ap.parse_args()
A, B, pt, w, meta = load_instance(a.case)
n = len(w)
rng = np.random.default_rng(5)
out = dict(case=a.case, T=2 ** a.logT, betas={})
for b in [float(x) for x in a.betas.split(',')]:
    r = []
    for start in ('random', 'witness'):
        p0 = rng.permutation(n) if start == 'random' else w.copy()
        s, _ = switch_count(A, B, p0, w, b, 2 ** a.logT, int(0.15 * n), int(0.5 * n), int(rng.integers(2 ** 31)))
        r.append(int(s))
    out['betas'][str(b)] = r
    print(a.case, b, r, flush=True)
json.dump(out, open(os.path.join(V2, 'results', 'gt', f'switch_{a.case}.json'), 'w'))
