"""R15 S1: structured-mass floors (all RIGOROUS lower bounds on Lambda(beta) = log sum_pi e^{beta S(pi)}):
  point : beta S(w)
  ball  : beta S(w) + log sum_{d_H(pi,w)<=2} e^{beta (S(pi)-S(w))}   (exact sum)
  coset : coset ELBO (gacert2.variational; K = 6; blocks from a C-restricted chain, 150 x 2^14 steps, seed 7;
          8 sweeps of block-coordinate ascent).  max(point, coset) is also a floor.
Proteins (22 pairs, cached witness, k0 = released k0): beta in beta_c + {0, 0.02, ..., 0.40} U {2.0, 3.0},
beta_c = mean of the released two-phase reps (TI-BASED; only used to place the grid).
Silent synthetic (16 pairs): the 4 evaluation betas, with (a) the graph-only witness, (b) the planted pi*.
Writes results/r15/mass_floor/<case>.json.  Comparison with FP-TI runs: r15_mass_floor_analyze.py.
Usage: python r15_mass_floor.py worker <wid>"""
import glob, json, os, sys, time
import numpy as np
from r15_common import RES, V3, prov, dump, claim, append_jsonl
from gacert2.instances import load_instance, synth_pair, find_witness
from gacert2.phase import log_W_sum
from gacert2.variational import chain_samples, make_blocks, coset_elbo
from gacert.core import score

OUT = os.path.join(RES, 'mass_floor')
TP = os.path.join(V3, 'results', 'twophase')
SG = os.path.join(V3, 'results', 'silent_graph')


def floors(A, B, w, betas, k0, seed=7):
    Sw = int(score(A, B, w))
    rows = []
    prop = 0
    for b in betas:
        t = time.time()
        samp, p = chain_samples(A, B, w, b, k0, 150, 2 ** 14, seed)
        prop += p
        blocks = make_blocks(w, samp, 6)
        r = coset_elbo(A, B, w, b, blocks, sweeps=8)
        rows.append(dict(beta=float(b), point=float(b * Sw), ball=float(b * Sw + log_W_sum(A, B, w, float(b))),
                         coset=r['elbo'], coset_ES=r['ES'], coset_H=r['H'], n_blocks_gt1=r['n_blocks_gt1'],
                         block_sizes=np.bincount([len(x) for x in blocks]).tolist(), seconds=time.time() - t))
    return Sw, rows, prop


def run_protein(case):
    path = os.path.join(OUT, case + '.json')
    if os.path.exists(path):
        return
    t0 = time.time()
    A, B, pt, w, meta = load_instance(case)
    d = json.load(open(os.path.join(TP, case + '_all.json')))
    bc = float(np.mean([r['beta_c'] for r in d['reps']]))
    betas = [round(bc + 0.02 * i, 6) for i in range(21)] + [2.0, 3.0]
    Sw, rows, prop = floors(A, B, w, betas, d['k0'])
    # TI-BASED context: s_C(beta) = F_C(beta) - beta S(w) from the released curves (mean over reps)
    for r in rows:
        v = []
        for rep in d['reps']:
            bC, FC = np.array(d['gridC']), np.array(rep['FC'])
            if bC[0] > bC[-1]:
                bC = bC[::-1]
            v.append(float(np.interp(r['beta'], bC, FC) - r['beta'] * Sw))
        r['s_C_TI_BASED'] = float(np.mean(v))
    rec = dict(case=case, kind='protein', n=len(w), Sw=Sw, k0=d['k0'], beta_c_TI_BASED=bc, rows=rows,
               chain_proposals=prop, config=dict(K=6, nsamp=150, T=2 ** 14, seed=7, sweeps=8))
    rec['prov'] = prov(__file__, t0)
    dump(rec, path)
    print(case, 'gain@bc+0.1', round(rows[5]['coset'] - rows[5]['point'], 2), 'ball', round(rows[5]['ball'] - rows[5]['point'], 2),
          f"{rec['prov']['wall_seconds']:.0f}s", flush=True)


def run_silent(tag):
    path = os.path.join(OUT, 'silent_' + tag + '.json')
    if os.path.exists(path):
        return
    t0 = time.time()
    s = json.load(open(os.path.join(SG, tag + '.json')))
    A, B, ps = synth_pair(s['n'], s['deg'], s['q'], s['seed'])
    w = find_witness(A, B, restarts=4)
    assert int(score(A, B, w)) == s['Sw']
    betas = [r['beta'] for r in s['rows']]
    out = dict(case='silent_' + tag, kind='silent', n=s['n'], k0=s['k0'], beta_c_ref=s['beta_c_GT'],
               witness_success=bool(s['Sw'] >= s['S_planted']))
    prop = 0
    for name, ww in (('graph_witness', w), ('planted', ps)):
        Sw, rows, p = floors(A, B, ww, betas, s['k0'])
        prop += p
        out[name] = dict(S=Sw, rows=rows)
    out['chain_proposals'] = prop
    out['prov'] = prov(__file__, t0)
    dump(out, path)
    print(tag, [round(r['coset'] - r['point'], 2) for r in out['graph_witness']['rows']],
          [round(r['coset'] - r['point'], 2) for r in out['planted']['rows']], flush=True)


if __name__ == '__main__':
    wid = sys.argv[2]
    cases = sorted((json.load(open(f))['n'], os.path.basename(f)[:-9]) for f in glob.glob(os.path.join(TP, '*_all.json')))
    jobs = [('p', c) for _, c in cases] + [('s', os.path.basename(f)[:-5]) for f in sorted(glob.glob(os.path.join(SG, '*.json')))]
    for k, c in jobs:
        if not claim(os.path.join(OUT, 'locks'), c):
            continue
        t = time.time()
        try:
            run_protein(c) if k == 'p' else run_silent(c)
            append_jsonl(dict(job=c, worker=wid, secs=time.time() - t, ok=True), os.path.join(OUT, 'jobs_done.jsonl'))
        except Exception as e:
            import traceback; traceback.print_exc()
            append_jsonl(dict(job=c, worker=wid, ok=False, err=repr(e)), os.path.join(OUT, 'jobs_done.jsonl'))
