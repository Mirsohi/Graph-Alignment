"""Small synthetic correlated graphs: pilot beta_c -> ground truth (long MH,
switch counts) at beta_c + offsets -> two-phase estimator -> baselines.
Usage: python synth_pipeline.py syn_n40_d6_q0.1_s0 [...]"""
import json, os, sys, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
sys.path.insert(0, HERE)
import run_twophase, gt_mh, baselines

OFF = [-0.08, -0.03, -0.015, 0.0, 0.015, 0.03, 0.08]
for case in sys.argv[1:]:
    pil = run_twophase.run(case, 2, 18, 0.3, [], 'c8', 6.0, 'pilot')
    bc = float(np.nanmean([r['beta_c'] for r in pil['reps']]))
    if not np.isfinite(bc):
        print(case, 'no transition found; skip', flush=True)
        continue
    betas = [round(bc + o, 3) for o in OFF]
    print(case, 'pilot beta_c', bc, betas, flush=True)
    gt_mh.run(case, betas, 28, 0.3, 0, os.path.join(V2, 'results', 'gt'))
    run_twophase.run(case, 4, 18, 0.3, betas, 'c8', 6.0, 'gt')
    baselines.run(case, betas, 26, 0.3)
