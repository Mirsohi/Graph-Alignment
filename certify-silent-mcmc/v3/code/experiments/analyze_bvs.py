"""Summary table for the BVS certificate experiments (all results/bvs/*.json)."""
import json, os, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
R = os.path.join(V2, 'results', 'bvs')
rows = []
agg = {}
for f in sorted(glob.glob(os.path.join(R, '*.json'))):
    d = json.load(open(f))
    import re
    fam = re.sub(r'_s\d+$', '', d['name'])
    ex = d.get('exact', {})
    r = dict(name=d['name'], p=d['p'], witness=d['witness']['source'], witness_is_truth=d['witness']['equals_truth'],
             ev_fp=d['evidence']['fp_ti'], ev_2ph=d['evidence']['two_phase'],
             ev_lb=d['evidence']['witness_lower_bound'], ev_exact=ex.get('log_evidence'),
             lam_max=d['ti']['lam_max_melt'])
    for s, c in d['chains'].items():
        r[f'{s}_cert'] = c['n_cert']
        r[f'{s}_rhat'] = (c['rhat_ell'], c['rhat_size'])
        r[f'{s}_truth_incl'] = float(np.mean([x['truth_incl'] for x in c['rows']]))
        a = agg.setdefault((fam, s), dict(chains=0, stuck=0, rhat_pass=0, groups=0,
                                          **{k: 0 for k in c['n_cert']}))
        a['chains'] += len(c['rows'])
        a['stuck'] += sum(x['truth_incl'] < 0.5 for x in c['rows']) if 'weak' not in fam else 0
        if 'incl_maxabs_err' in c:
            a.setdefault('max_incl_err_vs_exact', []).append(round(c['incl_maxabs_err'], 3))
        a['groups'] += 1
        a['rhat_pass'] += int(c['rhat_ell'] < 1.01 and c['rhat_size'] < 1.01)
        for k, v in c['n_cert'].items():
            a[k] += v
    if ex:
        r['false_alarm_iid'] = ex['false_alarm_iid']
        r['two_phase_err'] = ex['two_phase_max_abs_err']
    rows.append(r)
    print(f"{d['name']:20s} witness={r['witness']}({r['witness_is_truth']}) logZ: FP-TI {r['ev_fp']:.2f} "
          f"2-phase {r['ev_2ph']:.2f} exact {r['ev_exact']} witness-LB {r['ev_lb']:.2f} lam_max {r['lam_max']:.3f}")
print('\nfamily/start: chains, stuck(truth incl<.5), groups with Rhat(ell,size)<1.01, certified counts')
for (fam, s), a in sorted(agg.items()):
    print(f'{fam:15s} {s:7s}', a)
json.dump(dict(rows=rows, agg={f'{k[0]}|{k[1]}': v for k, v in agg.items()}),
          open(os.path.join(V2, 'results', 'summary_bvs.json'), 'w'), indent=1, default=str)
