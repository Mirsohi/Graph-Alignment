"""Single entry point (round 14): regenerate every paper table from the saved results and
print the two status summaries the paper reports, without re-running any sampler.

  python v3/code/experiments/reproduce_tables.py

Steps
 1. make_tables.py  (also runs make_silent.py and make_heldout_table.py): Tables 2-14 (paper numbering of round 15),
    including the exhaustive released status of all 22 protein pairs (tables/released.tex,
    results/summary_released_22.json), which applies the fixed retry policy (one two-seed
    attempt, at most one two-seed retry on the same output) to the saved survey,
    fresh-seed, middle-basin and retry results.
 2. analyze_r5.py: Tables 15-19, including every one of the 19 full-pipeline replicates
    (tables/runs.tex, results/summary_r5.json -> runs) with first attempt, retry and cost.
 3. Prints (a) the 22 released statuses with counts and (b) the 19 runs, and labels each
    detection statistic as RIGOROUS (lambda = 0 certificate, witness floor) or TI-BASED
    (tilted bound, two-phase occupancy error, two-phase estimates).
Re-running the samplers themselves is described per table in the README.
 4. r15_reproduce.py (round 15): summaries, tables and figures of Appendices F-G, main-text Table 4 and Figures 2-3."""
import json, os, runpy, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
R = os.path.join(V2, 'results')

for script in ('make_tables.py', 'analyze_r5.py'):
    print(f'== running {script}', flush=True)
    runpy.run_path(os.path.join(HERE, script), run_name='__main__')

rel = json.load(open(os.path.join(R, 'summary_released_22.json')))
print('\n== Released status of the 22 protein pairs (fixed retry policy)')
for c, d in rel['pairs'].items():
    print(f"  {c:24s} survey={d['survey']:28s} first attempt={d['first']:12s} final={d['final']}")
print('  counts (final):', dict(Counter(d['final'] for d in rel['pairs'].values())))
print('  counts (first attempt):', dict(Counter(d['first'] for d in rel['pairs'].values())))

runs = json.load(open(os.path.join(R, 'summary_r5.json')))['runs']
print(f'\n== {len(runs)} full-pipeline replicates (TI-based estimates; status from the heuristic screen)')
for r in runs:
    print(f"  {r['case']:24s} seed {r['seed']}  beta_c={r['beta_c']:.4f}  max|dw|={r['max_dw']:.3f}  "
          f"first={r['status']}  retry={r['retry_status'] or '--'}  final={r['final_status']}")
print('  counts (final):', dict(Counter(r['final_status'] for r in runs)))

print('\n== Statistic types used in the tables')
print('  RIGOROUS : lambda=0 certificate (Tables 3, 14 "rigorous cert."), witness floor (Table 2 "certified FP error"; Tables 20-25)')
print('  TI-BASED : tilted bound, two-phase occupancy error (Tables 3, 14 "TI-based"), two-phase estimates (Tables 13, 15-19)')
print('  SCREEN   : validity-check status (Tables 8-10, 17; held-out evaluation Tables 4, 26-31, EMPIRICAL); heuristic, not an accuracy guarantee')

# R15 additions (Agent 3): regenerate the R15 summaries, draft tables and figures if R15 results exist.
if os.path.isdir(os.path.join(R, 'r15')):
    print('\n== R15 (r15_reproduce.py)')
    runpy.run_path(os.path.join(HERE, 'r15_reproduce.py'), run_name='__main__')
