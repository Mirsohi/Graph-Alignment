"""R15 M2 secondary analyses (plan: reference-uncertain instances 'kept in every denominator and also analysed
separately'; instance-level summaries).  Uses the FROZEN metric function r15_p1_analyze.block on the per-run records
stored in summary_p1.json; nothing is re-estimated.  Writes results/r15/p1_heldout/summary_p1_subsets.json and
paper/tables/r15_p1_instances.tex.  Usage: python r15_p1_subsets.py"""
import json, os
import numpy as np
from r15_common import V3, Enc
from r15_p1_common import P1
from r15_p1_analyze import block, EPS, METHODS, TOLS


def crit(runs, comps):
    O0 = block(runs, [r['O0'] for r in runs], EPS)
    O2 = block(runs, [r['O2'] for r in runs], EPS)
    ctrl = [r for r in runs if r['fam'] in ('F1', 'F3')]
    out = dict(fpr_O2=O2['eps0.05']['fpr'], fpr_O0=O0['eps0.05']['fpr'], coverage_O2=O2['coverage'],
               coverage_O2_controls=(sum(r['O2'] for r in ctrl) / len(ctrl)) if ctrl else None,
               q90_O2=O2['e_returned']['q90'], n_returned_O2=O2['n_returned'])
    ret = [r for r in runs if r['O2']]
    if ret and comps:
        out['comparator_q90_same_runs'] = {m: float(np.quantile([comps[m][r['inst']] for r in ret if r['inst'] in comps[m]], .9))
                                           for m in comps if any(r['inst'] in comps[m] for r in ret)}
    return out


def main():
    S = json.load(open(os.path.join(P1, 'summary_p1.json')))
    runs = S['runs']
    comps = {m: {c['inst']: c['e'] for c in S['comparators'][m]['instances']} for m in S['comparators']}
    unc = set(S['reference']['uncertain'])
    out = {}
    for name, sel in (('reference_certain', lambda r: r['inst'] not in unc), ('reference_uncertain', lambda r: r['inst'] in unc),
                      ('controls_F1_F3', lambda r: r['fam'] in ('F1', 'F3')), ('hidden_F2_F4', lambda r: r['fam'] in ('F2', 'F4'))):
        R = [r for r in runs if sel(r)]
        insts = sorted(set(r['inst'] for r in R))
        d = dict(n_instances=len(insts), n_runs=len(R), ops={})
        for op in ['O0', 'O1', 'O2'] + [f'O2_t{t}' for t in TOLS]:
            d['ops'][op] = block(R, [r[op] for r in R], EPS) if R else None
        d['comparators'] = {}
        for m in comps:
            cr = [c for c in S['comparators'][m]['instances'] if c['inst'] in insts]
            d['comparators'][m] = block(cr, [True] * len(cr), EPS) if cr else None
        d['criteria_like'] = crit(R, {m: {i: e for i, e in comps[m].items() if i in insts} for m in comps}) if R else None
        out[name] = d
    inst = []
    for i in S['instances_pipeline']:
        rr = [r for r in runs if r['inst'] == i]
        inst.append(dict(inst=i, ref_uncertain=i in unc, max_se_ref=S['reference']['max_se_ref'][i],
                         status=[r['status'] for r in rr], e=[r['e'] for r in rr],
                         comparator_e={m: comps[m].get(i) for m in comps}))
    out['instances'] = inst
    json.dump(out, open(os.path.join(P1, 'summary_p1_subsets.json'), 'w'), cls=Enc, indent=1)
    L = [r'% R15 M2 instance-level draft (EMPIRICAL). e = max_beta |a_hat - a_ref|; * = reference-uncertain (max SE_ref > 0.02).',
         r'\begin{tabular}{lcccccc}', r'\toprule',
         r'instance & SE$_{\rm ref}$ & status (rep 1 / 2) & $e$ rep 1 & $e$ rep 2 & $e$ WL & $e$ PT \\', r'\midrule']
    f = lambda x: '--' if x is None else f'{x:.3f}'
    for d in inst:
        L.append(f"{d['inst'].replace('_', '-')}{'*' if d['ref_uncertain'] else ''} & {d['max_se_ref']:.3f} & {' / '.join(d['status'])} & "
                 f"{f(d['e'][0])} & {f(d['e'][1] if len(d['e']) > 1 else None)} & {f(d['comparator_e'].get('wl_ext'))} & "
                 f"{f(d['comparator_e'].get('pt_broad_w'))} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(V3, 'paper', 'tables', 'r15_p1_instances.tex'), 'w').write('\n'.join(L) + '\n')
    return out


if __name__ == '__main__':
    o = main()
    for k in ('reference_certain', 'reference_uncertain', 'controls_F1_F3', 'hidden_F2_F4'):
        print(k, o[k]['n_instances'], json.dumps(o[k]['criteria_like'], cls=Enc))
