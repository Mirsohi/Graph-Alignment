"""R15 X1 analysis (stretch, rule fixed before data): comparators with a second seed and a replicate-agreement
abstention.  A comparator abstains on an instance if max_{beta in G} |a_seed1 - a_seed2| > 0.05; otherwise it returns
the two-seed average a(beta), with error e = max_beta |a_avg - a_ref|; compute charged = both seeds' proposals.
Uses the frozen metric function r15_p1_analyze.block.  EMPIRICAL.  Writes results/r15/p1_heldout/summary_x1.json."""
import glob, json, os
import numpy as np
from r15_common import Enc
from r15_p1_common import P1, TEST
from r15_p1_analyze import block, EPS


def main():
    out = dict(rule='abstain iff max_beta |a_s1 - a_s2| > 0.05; return 2-seed average; charged 2B', methods={})
    for m in ('wl_ext', 'pt_broad_w'):
        rows = []
        for i in TEST:
            f1, f2 = os.path.join(P1, 'comp', f'{i}_{m}.json'), os.path.join(P1, 'comp_x1', f'{i}_{m}.json')
            fr = os.path.join(P1, 'ref', f'{i}.json')
            if not (os.path.exists(f1) and os.path.exists(f2)):
                continue
            c1, c2, r = json.load(open(f1)), json.load(open(f2)), json.load(open(fr))
            a1 = np.array([x['a'] for x in c1['rows']]); a2 = np.array([x['a'] for x in c2['rows']])
            aref = np.array([x['a_ref'] for x in r['rows']])
            agree = float(np.max(np.abs(a1 - a2)))
            rows.append(dict(inst=i, fam=i[:2], agree_gap=agree, returned=agree <= 0.05,
                             e=float(np.max(np.abs(0.5 * (a1 + a2) - aref))), e_seed1=float(np.max(np.abs(a1 - aref))),
                             e_seed2=float(np.max(np.abs(a2 - aref))),
                             cost=int(c1['charged_proposals'] + c2['charged_proposals']),
                             sec=float(c1['prov']['wall_seconds'] + c2['prov']['wall_seconds']),
                             ref_uncertain=r['reference_uncertain']))
        b = block(rows, [x['returned'] for x in rows], EPS) if rows else None
        if rows:
            b['per_family'] = {f: block([x for x in rows if x['fam'] == f], [x['returned'] for x in rows if x['fam'] == f], EPS)
                               for f in sorted(set(x['fam'] for x in rows))}
            b['seed2_alone'] = block([dict(x, e=x['e_seed2']) for x in rows], [True] * len(rows), EPS)
        out['methods'][m] = dict(summary=b, instances=rows)
    json.dump(out, open(os.path.join(P1, 'summary_x1.json'), 'w'), cls=Enc, indent=1)
    return out


if __name__ == '__main__':
    o = main()
    for m, d in o['methods'].items():
        s = d['summary']
        if s:
            print(m, 'n', s['n_runs'], 'coverage', s['coverage'], 'fpr', s['eps0.05']['fpr'], 'returned', s['n_returned'],
                  'q90', s['e_returned']['q90'], 'seed2 alone fpr', s['seed2_alone']['eps0.05']['fpr'])
