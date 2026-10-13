"""Round 10: bounded retry of the released policy on the 22-pair survey.
The first attempt for each pair is the survey seed plus, for the 8 survey passes, the
fresh-seed repeat (results/validity_multi); 12 pairs are 'unresolved' after it.  For each of
them the released two-seed check (gacert2.diagnose, frozen rules) is run ONCE more with two
fresh seeds at the survey's beta_c (w_hat = 0.5 there by definition, cached graph-only
witness); the estimator is not re-run.  The final status is 'pass (retry)' if the retry
passes, 'multi-phase' if the middle-basin rule fires in both retry seeds, else 'unresolved'.
Writes results/survey_retry/<case>.json.
Usage: python survey_retry.py [part nparts]"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.diagnose import diagnose

UNRESOLVED = ['pscdb_N_75', 'pscdb_N_261', 'pscdb_ID_55', 'pscdb_N_271', 'pscdb_B_38', 'pscdb_N_288',
              'pscdb_CD_21', 'adk_crystal_control', 'pscdb_N_82',          # unresolved in the survey
              'pscdb_B_44', 'pscdb_N_74', 'pscdb_N_217']                     # fail on the fresh-seed repeat
OUT = os.path.join(V2, 'results', 'survey_retry')
os.makedirs(OUT, exist_ok=True)
todo = UNRESOLVED if len(sys.argv) < 3 else UNRESOLVED[int(sys.argv[1])::int(sys.argv[2])]
for case in todo:
    path = os.path.join(OUT, case + '.json')
    if os.path.exists(path):
        continue
    sv = json.load(open(os.path.join(V2, 'results', 'validity', case + '_s1.json')))
    A, B, pt, w, meta = load_instance(case)
    t0 = time.time()
    d = diagnose(A, B, w, sv['k0'], sv['beta_c'], 0.5, seeds=(2001, 2002))
    status = 'pass (retry)' if d['status'] == 'pass' else d['status']
    rec = dict(case=case, n=len(w), k0=sv['k0'], beta_c=sv['beta_c'], retry_status=d['status'], final_status=status,
               retry_failed_rules=[[k for k, v in r['rules'].items() if not v] for r in d['runs']],
               runs=d['runs'], retry_proposals=int(sum(r['proposals'] for r in d['runs'])), retry_sec=time.time() - t0)
    json.dump(rec, open(path, 'w'), indent=1, default=float)
    print(case, status, rec['retry_failed_rules'], f"{rec['retry_sec']:.0f}s", flush=True)
