"""C-side two-start test at the two-phase beta_c: C-restricted chains started at
the witness and at a random state with overlap k0+1; compare mean S and mean k.
Usage: python validity_c.py tag"""
import json, os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.windows import _start_with_overlap
from gacert2.phase import rmh
tag = sys.argv[1]
out = {}
for f in sorted(glob.glob(os.path.join(V2, 'results', 'twophase', f'*_{tag}.json'))):
    d = json.load(open(f))
    case, n, k0 = d['case'], d['n'], d['k0']
    bc = float(np.nanmean([r['beta_c'] for r in d['reps']]))
    A, B, pt, w, meta = load_instance(case)
    rng = np.random.default_rng(5)
    res = {}
    T = 2 ** 23
    for nm, p0 in (('witness_start', w.copy()), ('partial_start', _start_with_overlap(w, k0 + 1, 11))):
        r = rmh(A, B, p0, w, bc, T, T // 4, 64, k0, n, int(rng.integers(2 ** 31)), False)
        st, kt = r[0], r[1]
        L = len(st) // 20
        res[nm] = dict(meanS=float(st.mean()), seS=float(st[:20 * L].reshape(20, L).mean(1).std(ddof=1) / np.sqrt(20)),
                       meank=float(kt.mean()))
    dS = abs(res['witness_start']['meanS'] - res['partial_start']['meanS'])
    se = np.hypot(res['witness_start']['seS'], res['partial_start']['seS'])
    res['ok'] = bool(dS < 3 * se + 2 and abs(res['witness_start']['meank'] - res['partial_start']['meank']) < 0.05 * n)
    res['dS'] = float(dS)
    out[case] = res
    print(f"{case:22s} bc={bc:.3f} C-melt dS={dS:.1f}±{se:.1f} k {res['witness_start']['meank']:.0f}/{res['partial_start']['meank']:.0f} ok={res['ok']}", flush=True)
json.dump(out, open(os.path.join(V2, 'results', f'validity_c_{tag}.json'), 'w'), indent=1)
