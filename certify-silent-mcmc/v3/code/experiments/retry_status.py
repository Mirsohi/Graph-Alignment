"""Round 10: the bounded retry of the released policy (gacert2.diagnose.diagnose_bounded)
applied to every full-pipeline replicate whose first two-seed attempt
(results/replicate_status/) was 'unresolved'.  The retry re-runs the two-seed check once,
with two fresh seeds (300 + seed, 400 + seed), on the same estimator output (the run's own
beta_c, w_hat = 0.5, the cached graph-only witness); the estimator is not re-run.
Replicates whose first attempt passed are not retried.
Writes results/retry_status/<case>_s<seed>.json and results/retry_status/summary.json.
Usage: python retry_status.py"""
import json, os, sys, glob, time
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.diagnose import diagnose

OUT = os.path.join(V2, 'results', 'retry_status')
os.makedirs(OUT, exist_ok=True)
todo = []
for f in sorted(glob.glob(os.path.join(V2, 'results', 'replicate_status', '*.json'))):
    r = json.load(open(f))
    if r['status'] == 'unresolved':
        todo.append((f, r))
if len(sys.argv) > 1:
    todo = todo[int(sys.argv[1])::int(sys.argv[2])]
for f, r in todo:
    path = os.path.join(OUT, os.path.basename(f))
    if os.path.exists(path):
        continue
    A, B, pt, w, meta = load_instance(r['case'])
    rp = json.load(open(os.path.join(V2, 'results', 'pipeline_reps', os.path.basename(f))))
    t0 = time.time()
    d = diagnose(A, B, w, rp['k0'], r['beta_c'], 0.5, seeds=(300 + r['seed'], 400 + r['seed']))
    status = 'pass (retry)' if d['status'] == 'pass' else d['status']
    rec = dict(case=r['case'], seed=r['seed'], beta_c=r['beta_c'], first_status=r['status'],
               first_failed_rules=r['failed_rules'], retry_status=d['status'], final_status=status,
               retry_failed_rules=[[k for k, v in x['rules'].items() if not v] for x in d['runs']],
               runs=d['runs'], retry_proposals=int(sum(x['proposals'] for x in d['runs'])),
               retry_sec=time.time() - t0)
    json.dump(rec, open(path, 'w'), indent=1, default=float)
    print(r['case'], r['seed'], status, rec['retry_failed_rules'], f"{rec['retry_sec']:.0f}s", flush=True)
