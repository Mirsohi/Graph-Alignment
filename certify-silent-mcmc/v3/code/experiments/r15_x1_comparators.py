"""R15 X1 (stretch): second comparator seed and a replicate-agreement abstention for the M2 comparators.
Identical to the frozen r15_p1_comparators.run except the seed: default_rng(6151 + idx) instead of 5151 + idx
(same budget B, same witness, same ladder / WL settings).  Outputs results/r15/p1_heldout/comp_x1/<inst>_<method>.json.
Rule (fixed before any X1 data, logged in notes/R15_PROGRESS.md): a comparator ABSTAINS on an instance if
max_{beta in G} |a_seed1 - a_seed2| > 0.05; otherwise it returns the average of the two seeds' a(beta);
its compute is charged at 2B.  Analysis: r15_x1_analyze.py.
Usage: python r15_x1_comparators.py worker <wid>"""
import os, sys, time
import numpy as np
import r15_p1_comparators as C
from r15_common import claim, append_jsonl
from r15_p1_common import P1, TEST

OUT = os.path.join(P1, 'comp_x1')


class _Rng:
    """default_rng shim: the frozen module calls np.random.default_rng(5151 + idx); map it to 6151 + idx."""
    def __init__(self, real):
        self.real = real

    def __call__(self, seed=None, *a, **k):
        if isinstance(seed, (int, np.integer)) and 5151 <= int(seed) < 5251:
            seed = int(seed) + 1000
        return self.real(seed, *a, **k)


def run(inst, method, outdir=OUT):
    real, dump0 = np.random.default_rng, C.dump
    C.np.random.default_rng = _Rng(real)
    C.dump = lambda obj, path: dump0({**obj, 'seed': obj['seed'] + 1000, 'x1': 'second comparator seed (6151+idx) via '
                                      'r15_x1_comparators.py'}, path)
    try:
        return C.run(inst, method, outdir=outdir)
    finally:
        C.np.random.default_rng, C.dump = real, dump0


if __name__ == '__main__':
    wid = sys.argv[2]
    for i in TEST:
        for m in ('wl_ext', 'pt_broad_w'):
            nm = f'{i}_{m}'
            if os.path.exists(os.path.join(OUT, nm + '.json')) or not claim(os.path.join(OUT, 'locks'), nm):
                continue
            t = time.time()
            try:
                run(i, m)
                append_jsonl(dict(job=nm, worker=wid, secs=time.time() - t, ok=True), os.path.join(OUT, 'jobs_done.jsonl'))
            except Exception as e:
                import traceback; traceback.print_exc()
                append_jsonl(dict(job=nm, worker=wid, ok=False, err=repr(e)), os.path.join(OUT, 'jobs_done.jsonl'))
