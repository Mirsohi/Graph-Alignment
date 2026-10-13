"""Round 10: released two-seed diagnostic (gacert2.diagnose, frozen rules at beta_c)
on the five pairs labelled multi-phase in the one-seed survey, to test whether their
label satisfies the released predicate (middle-basin rule fires in both seeds).
beta_c is the survey's two-phase estimate (results/validity/<case>_s1.json); w_hat = 0.5
there by definition; the witness is the cached graph-only witness.  Two fresh seeds.
Writes results/middle_recheck/<case>.json.
Usage: python middle_recheck.py [case ...]"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.diagnose import diagnose
from gacert.core import score

CASES = ['pscdb_O_16', 'pscdb_N_185', 'pscdb_IL_55', 'pscdb_IL_74', 'adk_open_closed']
OUT = os.path.join(V2, 'results', 'middle_recheck')
os.makedirs(OUT, exist_ok=True)
for case in (sys.argv[1:] or CASES):
    path = os.path.join(OUT, case + '.json')
    if os.path.exists(path):
        continue
    sv = json.load(open(os.path.join(V2, 'results', 'validity', case + '_s1.json')))
    A, B, pt, w, meta = load_instance(case)
    t0 = time.time()
    d = diagnose(A, B, w, sv['k0'], sv['beta_c'], 0.5, seeds=(1001, 1002))
    n = len(w)
    rec = dict(case=case, n=n, k0=sv['k0'], beta_c=sv['beta_c'], witness_score=int(score(A, B, w)),
               status=d['status'],
               failed_rules=[[k for k, v in r['rules'].items() if not v] for r in d['runs']],
               middle_basin_fired=[not r['rules']['middle_basin'] for r in d['runs']],
               middle_window=[0.1 * n, sv['k0'] + 0.05 * n],
               survey_basins=[b for b in sv['basins'] if b['mass'] >= 0.01],
               runs=d['runs'], diag_proposals=int(sum(r['proposals'] for r in d['runs'])),
               diag_sec=time.time() - t0)
    json.dump(rec, open(path, 'w'), indent=1, default=float)
    print(case, rec['status'], rec['failed_rules'], f"{rec['diag_sec']:.0f}s", flush=True)
