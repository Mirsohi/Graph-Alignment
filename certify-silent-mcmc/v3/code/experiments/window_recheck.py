"""Round-6 re-check of the window-profile rule at beta_c with the second-pass budget
(2^22 steps per window, two fresh seeds), for pairs whose fresh 2^21 window check at
beta_c failed in validity_multi.py.  Same rule as the original protocol: the pair
passes on the second window pass if both 2^22 profiles satisfy |dlogit| < 1.
Usage: python window_recheck.py case1,case2"""
import json, os, sys
import numpy as np
from scipy.special import logsumexp
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.windows import overlap_profile
out = {}
for case in sys.argv[1].split(','):
    v = json.load(open(os.path.join(V2, 'results', 'validity_multi', case + '.json')))
    row = [r for r in v['rows'] if r['offset'] == 0.0][0]
    A, B, pt, w, meta = load_instance(case)
    k0 = v['k0']
    ws = []
    for s in (31, 32):
        lp = overlap_profile(A, B, w, row['beta'], 2 ** 22, s)['logp']
        f = np.isfinite(lp[k0:])
        ws.append(float(np.exp(logsumexp(lp[k0:][f]))))
    lg = lambda x: np.log(x / (1 - x))
    ok = [bool(abs(lg(x) - lg(row['w_est'])) < 1) for x in ws]
    out[case] = dict(beta=row['beta'], w_est=row['w_est'], w_window_2e21_fresh=row['w_window'], w_window_2e22=ws, ok=ok,
                     second_pass=bool(all(ok)))
    print(case, out[case], flush=True)
json.dump(out, open(os.path.join(V2, 'results', 'validity_multi', 'window_recheck.json'), 'w'), indent=1)
