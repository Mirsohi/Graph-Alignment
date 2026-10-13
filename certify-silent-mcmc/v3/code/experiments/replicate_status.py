"""Per-replicate diagnostic status (round 7): the released default diagnostic
(gacert2.diagnose: transition-only frozen rules, two independent seeds) evaluated at
each full-pipeline replicate's own beta_c (w_hat = 0.5 there by definition), with the
graph-only witness of that instance.  Every replicate reached the cached witness score;
the cached witness permutation is used.  Edge occupancy is re-measured with dedicated
restricted chains because the replicate files do not store the TI chains.
Writes results/replicate_status/<case>_s<seed>.json.
Usage: python replicate_status.py"""
import json, os, sys, glob, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.diagnose import diagnose

OUT = os.path.join(V2, 'results', 'replicate_status')
os.makedirs(OUT, exist_ok=True)
for f in sorted(glob.glob(os.path.join(V2, 'results', 'pipeline_reps', '*.json')), reverse=len(sys.argv) > 1):
    rp = json.load(open(f))
    path = os.path.join(OUT, os.path.basename(f))
    if os.path.exists(path):
        continue
    t0 = time.time()
    A, B, pt, w, meta = load_instance(rp['case'])
    k0 = rp['k0']
    d = diagnose(A, B, w, k0, rp['beta_c'], 0.5, seeds=(100 + rp['seed'], 200 + rp['seed']))
    rec = dict(case=rp['case'], seed=rp['seed'], beta_c=rp['beta_c'], witness_score=rp['Sw'], witness_score_best=rp['Sw_cached'],
               status=d['status'], failed_rules=[[k for k, v in r['rules'].items() if not v] for r in d['runs']],
               runs=d['runs'], diag_proposals=int(sum(r['proposals'] for r in d['runs'])), diag_sec=time.time() - t0)
    json.dump(rec, open(path, 'w'), indent=1, default=float)
    print(rp['case'], rp['seed'], rec['status'], rec['failed_rules'], f"{rec['diag_sec']:.0f}s", flush=True)
