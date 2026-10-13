"""R15 S2: predictive check of the delay law (Prop. 1(d)) on the M1 FP-TI runs (0 CPU-h).
For each released two-phase rep (TI-BASED curves; results/twophase/<case>_all.json):
  s_C = F_C(beta_c) - beta_c S(w),  c_D = log n! - log|D|,
  delta_lo = (s_C + c_D) / (S(w) - E_D S(beta_c)),
  delta_hi = (s_C + c_D) / (S(w) - E_D S(beta_det)),  beta_det = first beta with beta S(w) - F_D(beta) - c_D > 0
(as notes/r15_prototypes/delay.py); per pair, beta_c, delta_lo, delta_hi are averaged over the released reps.
Prediction for each M1 protein run: 'certified before nucleation' iff beta_nuc > beta_c + delta_lo.
Observed: first certified beta (rigorous point floor) exists and is < beta_nuc  (beta_nuc = first grid beta with
mean overlap > 0.3 n).  Agreement and every misclassification are reported; < 80% agreement -> keep (d) idealised.
Usage: python r15_delay_check.py -> results/r15/summary_delay.json"""
import glob, json, os
import numpy as np
from scipy.special import gammaln
from r15_common import RES, V3, Enc

TP = os.path.join(V3, 'results', 'twophase')


def law(case):
    d = json.load(open(os.path.join(TP, case + '_all.json')))
    n, Sw = d['n'], d['meta']['Sw']
    out = []
    for r in d['reps']:
        bD, FD = np.array(d['gridD']), np.array(r['FD'])
        bC, FC = np.array(d['gridC']), np.array(r['FC'])
        if bC[0] > bC[-1]:
            bC = bC[::-1]
        bc = r['beta_c']
        cD = gammaln(n + 1) - r['logD']
        sC = float(np.interp(bc, bC, FC) - bc * Sw)
        g = np.arange(bc, bD[-1], 0.001)
        f = g * Sw - np.interp(g, bD, FD) - cD
        i = np.where(f > 0)[0]
        bdet = float(g[i[0]]) if len(i) else float('nan')
        EDc = float(np.interp(bc, bD, r['meanS_D']))
        EDd = float(np.interp(bdet, bD, r['meanS_D'])) if np.isfinite(bdet) else float('nan')
        out.append(dict(beta_c=bc, s_C=sC, c_D=float(cD), delta_lo=(sC + cD) / (Sw - EDc),
                        delta_hi=(sC + cD) / (Sw - EDd) if np.isfinite(EDd) else float('nan'),
                        delay_idealised=bdet - bc))
    return dict(reps=out, beta_c=float(np.mean([o['beta_c'] for o in out])),
                delta_lo=float(np.mean([o['delta_lo'] for o in out])),
                delta_hi=float(np.nanmean([o['delta_hi'] for o in out])),
                s_C=float(np.mean([o['s_C'] for o in out])))


def main():
    runs = []
    laws = {}
    for f in sorted(glob.glob(os.path.join(RES, 'fpti_cert', '*.json'))):
        r = json.load(open(f))
        if r.get('info', {}).get('kind') != 'protein' or r['seed'] in (103, 104):   # M1 runs only (X2 seeds excluded)
            continue
        c = r['case']
        if c not in laws:
            laws[c] = law(c)
        L = laws[c]
        nuc, fc = r['nucleation_beta'], r['RIGOROUS']['first_cert_beta_point']
        pred = bool(nuc is not None and nuc > L['beta_c'] + L['delta_lo']) if nuc is not None else True
        obs = bool(fc is not None and (nuc is None or fc < nuc))
        runs.append(dict(case=c, logT=r['logT_per_point'], seed=r['seed'], nucleation=nuc, first_cert=fc,
                         beta_c_TI_BASED=L['beta_c'], delta_lo=L['delta_lo'], delta_hi=L['delta_hi'],
                         predicted=pred, observed=obs, agree=pred == obs,
                         ambiguous_zone=bool(nuc is not None and L['beta_c'] + L['delta_lo'] < nuc <= L['beta_c'] + L['delta_hi'])))
    n = len(runs)
    agree = sum(r['agree'] for r in runs)
    out = dict(note='delta_lo/hi and beta_c are TI-BASED (released two-phase curves); first_cert is RIGOROUS (point floor). '
                    'Predictive check, not validation.',
               n_runs=n, agree=agree, agreement=(agree / n if n else None), passes_80pct=bool(n and agree / n >= 0.8),
               by_budget={f'T{t}': dict(n=sum(r['logT'] == t for r in runs), agree=sum(r['agree'] for r in runs if r['logT'] == t))
                          for t in sorted(set(r['logT'] for r in runs))},
               confusion=dict(tp=sum(r['predicted'] and r['observed'] for r in runs),
                              fp=sum(r['predicted'] and not r['observed'] for r in runs),
                              fn=sum(not r['predicted'] and r['observed'] for r in runs),
                              tn=sum(not r['predicted'] and not r['observed'] for r in runs)),
               misclassified=[r for r in runs if not r['agree']], laws=laws, runs=runs)
    json.dump(out, open(os.path.join(RES, 'summary_delay.json'), 'w'), cls=Enc, indent=1)
    return out


if __name__ == '__main__':
    o = main()
    print(json.dumps({k: o[k] for k in ('n_runs', 'agree', 'agreement', 'passes_80pct', 'by_budget', 'confusion')}, cls=Enc))
