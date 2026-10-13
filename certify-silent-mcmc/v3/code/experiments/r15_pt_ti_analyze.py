"""R15 S3 analysis: witness floor (RIGOROUS) vs PT-based TI (2^29 proposals, 64 rungs on [0, 3], seed 1).
Writes results/r15/summary_pt_ti.json and paper/tables/r15_pt_ti.tex.  Either outcome is reportable; a silent
floor means PT-based TI was not certifiably low, NOT a failure of the certificate."""
import glob, json, os
import numpy as np
from r15_common import RES, V3, Enc

REL = json.load(open(os.path.join(V3, 'results', 'summary_released_22.json')))['pairs']


def main():
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, 'pt_ti', '*.json')))]
    recs = [r for r in recs if 'RIGOROUS' in r]
    fp = {}
    for f in glob.glob(os.path.join(RES, 'fpti_cert', '*_T21_s*.json')):
        r = json.load(open(f))
        if r['info']['kind'] == 'protein' and r['seed'] in (101, 102):
            fp.setdefault(r['case'], []).append(next(e['E_cert_point'] for e in r['RIGOROUS']['eval'] if e['beta'] == 3.0))
    per = {}
    for r in recs:
        e3 = r['RIGOROUS']['eval']['3.0']['E_cert_point']
        per[r['case']] = dict(n=r['n'], E3=e3, E25=r['RIGOROUS']['eval']['2.5']['E_cert_point'],
                              E20=r['RIGOROUS']['eval']['2.0']['E_cert_point'], first_cert=r['RIGOROUS']['first_cert_beta'],
                              best_S=r['best_S'], Sw=r['Sw_cached'], S_star_source=r['S_star_source'],
                              swap_min=r['swap_min'], swap_median=r['swap_median'], seconds=r['prov']['wall_seconds'],
                              proposals=r['proposals'], fpti_T21_E3=fp.get(r['case']), released_status=REL[r['case']]['final'])
    E = [p['E3'] for p in per.values()]
    out = dict(note='RIGOROUS witness floor vs PT-based TI (2^29 total proposals, 64 rungs uniform on [0,3], seed 1)',
               n_pairs=len(per), certified_at3=int(sum(e > 0 for e in E)),
               certified_at25=int(sum(p['E25'] > 0 for p in per.values())),
               certified_at20=int(sum(p['E20'] > 0 for p in per.values())),
               median_E3=float(np.median(E)) if E else None, max_E3=float(max(E)) if E else None,
               min_E3=float(min(E)) if E else None,
               pt_found_better_than_witness=[c for c, p in per.items() if p['best_S'] > p['Sw']],
               pt_best_below_witness=int(sum(p['best_S'] < p['Sw'] for p in per.values())),
               compute=dict(total_proposals=int(sum(p['proposals'] for p in per.values())),
                            total_seconds=float(sum(p['seconds'] for p in per.values()))),
               pairs=per)
    json.dump(out, open(os.path.join(RES, 'summary_pt_ti.json'), 'w'), cls=Enc, indent=1)
    if per:
        L = [r'% R15 S3 draft (r15_pt_ti_analyze.py). RIGOROUS witness floor vs PT-based TI (one seed).',
             r'\begin{tabular}{lcccc}', r'\toprule',
             r'TI method (total proposals) & certified at $\beta{=}3$ & median ${\rm Err}_{\rm cert}(3)$ & max ${\rm Err}_{\rm cert}(3)$ & certified at $\beta{=}2$ \\',
             r'\midrule',
             f"PT-TI, 64 rungs ($2^{{29}}$) & {out['certified_at3']}/{out['n_pairs']} & {out['median_E3']:.1f} & {out['max_E3']:.1f} & "
             f"{out['certified_at20']}/{out['n_pairs']} \\\\", r'\bottomrule', r'\end{tabular}']
        open(os.path.join(V3, 'paper', 'tables', 'r15_pt_ti.tex'), 'w').write('\n'.join(L) + '\n')
    return out


if __name__ == '__main__':
    o = main()
    print(json.dumps({k: v for k, v in o.items() if k != 'pairs'}, cls=Enc, indent=1))
