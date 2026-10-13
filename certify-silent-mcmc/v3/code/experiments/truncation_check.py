"""Size of the truncated-prior anchor correction at p = 200 (round 6): binomial tails
Pr[Bin(p,h) > m] and Pr[Bin(p-r,h) > m-r] for every p = 200 instance (m = 60, r = |w|).
Writes results/truncation_anchor_check.json."""
import json, os, glob
from scipy.stats import binom
V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = {}
for f in sorted(glob.glob(os.path.join(V2, 'results', 'bvs', '*200*.json'))):
    d = json.load(open(f))
    p, h = d['p'], d['h']
    r = d['candidates'][d['witness']['source']]['size']
    out[d['name']] = dict(r=r, tail_all=float(binom.sf(60, p, h)), tail_C=float(binom.sf(60 - r, p - r, h)))
json.dump(out, open(os.path.join(V2, 'results', 'truncation_anchor_check.json'), 'w'), indent=1)
print(max(max(v['tail_all'], v['tail_C']) for v in out.values()))
