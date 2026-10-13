"""All 22 protein pairs: two-phase beta_c vs v1 heuristic beta-hat_c vs the v1
hysteresis window; estimator spinodals; bidirectional certificates applied to
the v1 chains (per chain, using its posterior mass on the known alignment as
proxy for its phase).  Writes v2/results/summary_all.json.
Usage: python analyze_all.py [tag]"""
import json, os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(V2)
R = os.path.join(V2, 'results')
tag = sys.argv[1] if len(sys.argv) > 1 else 'all'
v1sum = json.load(open(os.path.join(ROOT, 'results', 'summary_protein.json')))


def spinodals(d):
    """D-spinodal: first beta on the D path where >5% of retained draws sit on the
    region edge (the diffuse branch stops being metastable).  C-spinodal: last beta
    (going down) before >5% of C draws sit on the edge."""
    gD, gC = np.array(d['gridD']), np.array(d['gridC'])
    sD, sC = [], []
    for r in d['reps']:
        eD, eC = np.array(r['edge_D']), np.array(r['edge_C'])
        iD = np.flatnonzero(eD > 0.05)
        sD.append(float(gD[iD[0]]) if len(iD) else np.nan)
        iC = np.flatnonzero(eC > 0.05)
        sC.append(float(gC[iC[-1]]) if len(iC) else np.nan)
    return float(np.nanmean(sD)), float(np.nanmean(sC))


def wcond(d, b):
    out = []
    gD, gC = np.array(d['gridD']), np.array(d['gridC'])
    for r in d['reps']:
        fd = np.interp(b, gD, r['FD']) if b <= gD[-1] else np.nan
        fc = np.interp(b, gC, r['FC'])
        if np.isnan(fd):   # beyond D grid: D is past its spinodal; use last slope (upper-ish)
            fd = r['FD'][-1] + (b - gD[-1]) * r['meanS_D'][-1]
        out.append((fc - fd))
    return float(np.mean(out)), float(np.std(out))


rows = []
cert = dict(diffuse_certified=0, diffuse_total=0, cond_certified=0, cond_total=0, by_beta={})
for f in sorted(glob.glob(os.path.join(R, 'twophase', f'*_{tag}.json'))):
    d = json.load(open(f))
    case = d['case']
    v1 = json.load(open(os.path.join(ROOT, 'results', 'protein', case + '.json')))
    bcs = [r['beta_c'] for r in d['reps']]
    sD, sC = spinodals(d)
    grp = {(g['label'], g['beta']): g for g in v1['groups'] if g['label'] in ('mh_uniform', 'mh_oracle')}
    betas = sorted({b for (_, b) in grp})
    melt = [b for b in betas if grp[('mh_oracle', b)]['match'] >= 0.5]
    freeze = [b for b in betas if grp[('mh_uniform', b)]['match'] >= 0.5]
    row = dict(case=case, n=d['n'], beta_c=float(np.mean(bcs)), beta_c_sd=float(np.std(bcs)),
               beta_c_v1=v1sum['beta_c'][case], spinodal_D=sD, spinodal_C=sC,
               v1_oracle_condensed_from=min(melt) if melt else None,
               v1_random_condensed_from=min(freeze) if freeze else None,
               fracW=[r['fracW_hi'] for r in d['reps']], Sw=d['meta']['Sw'], S_true=d['meta']['S_true'],
               witness_agree_true=d['meta']['witness_agree_true'],
               bestC=[r['best_C'] for r in d['reps']], proposals=d['reps'][0].get('proposals'))
    # bidirectional certificates on v1 chains (phase proxy: chain match >= 0.3 -> condensed)
    row['v1_chain_certs'] = {}
    for (lab, b), g in grp.items():
        if b > d['gridD'][-1]:
            continue        # no extrapolation of F_D beyond the TI grid (tangent would be anti-conservative)
        dF, sdF = wcond(d, b)
        wc = 1 / (1 + np.exp(-dF))
        nd = nc = cd = cc = 0
        for mch in g['match_chain']:
            if mch < 0.3:
                nd += 1
                cd += (1 - (1 - wc)) >= 0.99      # TV >= mu(C) when all draws in D
            else:
                nc += 1
                cc += (1 - wc) >= 0.99            # TV >= mu(D) when all draws in C
        row['v1_chain_certs'][f'{lab}_{b}'] = dict(dF=dF, sdF=sdF, w_cond=float(wc), n_diffuse=nd,
                                                   n_diffuse_cert=int(cd), n_cond=nc, n_cond_cert=int(cc))
        key = f'{lab}_{b}'
        bb = cert['by_beta'].setdefault(key, [0, 0, 0, 0])
        bb[0] += nd; bb[1] += int(cd); bb[2] += nc; bb[3] += int(cc)
    rows.append(row)
    print(f"{case:22s} n={d['n']:3d} bc={row['beta_c']:.3f}±{row['beta_c_sd']:.3f} v1={row['beta_c_v1']:.3f} "
          f"spinD={sD:.2f} spinC={sC:.2f} v1 oracle-cond>={row['v1_oracle_condensed_from']} "
          f"v1 rand-cond>={row['v1_random_condensed_from']} fracW={np.mean(row['fracW']):.2f} "
          f"bestC={max(row['bestC'])} Sw={row['Sw']}")
bc = np.array([r['beta_c'] for r in rows])
v1 = np.array([r['beta_c_v1'] for r in rows])
summ = dict(n_cases=len(rows), beta_c_range=[float(bc.min()), float(bc.max()), float(np.median(bc))],
            v1_minus_new=dict(median=float(np.median(v1 - bc)), min=float((v1 - bc).min()), max=float((v1 - bc).max())),
            inside_hysteresis=int(sum(1 for r in rows if r['spinodal_C'] <= r['beta_c'] <= r['spinodal_D'])),
            cert_by_group=cert['by_beta'], rows=rows)
print(json.dumps({k: v for k, v in summ.items() if k not in ('rows', 'cert_by_group')}, indent=1))
for k in sorted(cert['by_beta'], key=lambda s: (s.split('_')[1], float(s.split('_')[-1]))):
    print(k, 'diffuse chains (certified)', cert['by_beta'][k][:2], 'condensed chains (certified)', cert['by_beta'][k][2:])
json.dump(summ, open(os.path.join(R, f'summary_{tag}.json'), 'w'), indent=1)

# ---------------------------------------------------------------- I6 table
# Label every v1 local-MH chain (beta <= TI grid max) as CORRECT / WRONG phase using
# the two-phase weights (dominant phase with posterior weight >= 0.99; otherwise
# 'ambiguous', excluded).  Then tabulate alarm rates of standard diagnostics and of
# v1 certificates on correct vs wrong chains.  NOTE: labels come from our
# estimator (validated against ground truth on 3 pairs), not from ground truth.
sys.path.insert(0, os.path.join(ROOT, 'code', 'experiments'))
import analyze as v1a   # v1 analysis helpers (read-only use)
tab = {}
for row in rows:
    rec = v1a.enrich(json.load(open(os.path.join(ROOT, 'results', 'protein', row['case'] + '.json'))))
    for g in rec['groups']:
        if g['label'] not in ('mh_uniform', 'mh_oracle'):
            continue
        key = f"{g['label']}_{g['beta']}"
        if key not in row['v1_chain_certs']:
            continue
        wc = row['v1_chain_certs'][key]['w_cond']
        if 0.01 < wc < 0.99:
            continue
        dom = 'C' if wc >= 0.99 else 'D'
        rh_alarm = g['rhat'] >= 1.01
        std_alarm = not g['std_pass']
        for k, mch in enumerate(g['match_chain']):
            ph = 'C' if mch >= 0.3 else 'D'
            lab = 'correct' if ph == dom else 'wrong'
            t = tab.setdefault(lab, dict(chains=0, rhat_alarm=0, std_alarm=0, v1_tilted=0, v1_crude=0,
                                         bidir=0, by_start={}))
            t['chains'] += 1
            t['rhat_alarm'] += int(rh_alarm)
            t['std_alarm'] += int(std_alarm)
            t['v1_tilted'] += int(g['cert_chain_graph'][k]['tilted_cert'])
            t['v1_crude'] += int(g['cert_chain_graph'][k]['crude_cert'])
            t['bidir'] += int(lab == 'wrong')   # by construction (same labels): reported for completeness
            bs = t['by_start'].setdefault(g['label'], 0)
            t['by_start'][g['label']] = bs + 1
print('\nI6 (estimator-labelled) alarm rates:')
for lab, t in tab.items():
    c = t['chains']
    print(lab, c, {k: (v, round(v / c, 3)) for k, v in t.items() if k not in ('chains', 'by_start')}, t['by_start'])
summ['i6_table'] = tab
json.dump(summ, open(os.path.join(R, f'summary_{tag}.json'), 'w'), indent=1)
