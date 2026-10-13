"""R15 S1 analysis: compares the three RIGOROUS floors (point, transposition ball, coset ELBO) with every M1
FP-TI run, applies the pre-registered decision rule (operationalised in notes/R15_PROGRESS.md before data):
  (a) median over pairs of [coset - point] at beta_c + 0.10 >= 3 nats;
  (b1) pairs certified at beta=3 on all 2^21 seeds by max(point, coset) minus by point >= 2;
  (b2) over 2^21 runs certified by the point floor before nucleation, median of 1 - delay_coset/delay_point >= 0.30
       (delay = first certified beta on the S1 grid - beta_c);
  main text iff (a) and (b1 or b2).
Also: silent synthetic pairs (graph witness vs planted pi*) against the M1 synthetic FP-TI runs, and the per-chain
lambda=0 test of the silent table re-done with the coset floor (graph witness only).
beta_c and s_C are TI-BASED (released two-phase curves), used only for placement and context.
Writes results/r15/summary_mass_floor.json and paper/tables/r15_mass_floor.tex."""
import glob, json, os
import numpy as np
from scipy.special import gammaln
from r15_common import RES, V3, Enc

MF = os.path.join(RES, 'mass_floor')
FP = os.path.join(RES, 'fpti_cert')
LN01 = np.log(0.01)


def lam_at(path, b):
    """Lambda_FP at beta b: trapezoid integral continued with the linearly interpolated mean score."""
    x, m, L = np.array(path['beta']), np.array(path['meanS']), np.array(path['Lambda_FP'])
    i = int(np.clip(np.searchsorted(x, b, side='right') - 1, 0, len(x) - 2))
    mb = np.interp(b, x, m)
    return float(L[i] + 0.5 * (m[i] + mb) * (b - x[i]))


def med(v):
    return float(np.median(v)) if len(v) else None


def first(betas, E):
    for b, e in zip(betas, E):
        if e > 0:
            return b
    return None


def protein():
    pairs = {}
    for f in sorted(glob.glob(os.path.join(MF, 'pscdb_*.json')) + glob.glob(os.path.join(MF, 'adk_*.json'))):
        r = json.load(open(f))
        pairs[r['case']] = r
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(FP, '*.json')))]
    runs = [r for r in runs if r.get('info', {}).get('kind') == 'protein' and r['case'] in pairs
            and r['seed'] not in (103, 104)]   # M1 runs only (X2 seeds excluded)
    per_pair, per_run = {}, []
    for c, P in pairs.items():
        rows = P['rows']
        bc = P['beta_c_TI_BASED']
        g = {round(r['beta'] - bc, 2): r for r in rows[:21]}
        per_pair[c] = dict(n=P['n'], beta_c_TI_BASED=bc,
                           gain_coset=[r['coset'] - r['point'] for r in rows], gain_ball=[r['ball'] - r['point'] for r in rows],
                           betas=[r['beta'] for r in rows], s_C_TI_BASED=[r['s_C_TI_BASED'] for r in rows],
                           gain_coset_bc01=g[0.1]['coset'] - g[0.1]['point'], gain_ball_bc01=g[0.1]['ball'] - g[0.1]['point'],
                           gain_coset_3=rows[-1]['coset'] - rows[-1]['point'],
                           frac_of_sC_bc01_TI_BASED=((g[0.1]['coset'] - g[0.1]['point']) / g[0.1]['s_C_TI_BASED']
                                                     if g[0.1]['s_C_TI_BASED'] > 0 else None))
    for r in runs:
        P = pairs[r['case']]
        bets = [x['beta'] for x in P['rows']]
        L = [lam_at(r['path'], b) for b in bets]
        E = {k: [x[k] - l for x, l in zip(P['rows'], L)] for k in ('point', 'ball', 'coset')}
        E['best'] = [max(a, b, c) for a, b, c in zip(E['point'], E['ball'], E['coset'])]
        E['point_coset'] = [max(a, b) for a, b in zip(E['point'], E['coset'])]
        grid = bets[:21]
        fc = {k: first(grid, v[:21]) for k, v in E.items()}
        per_run.append(dict(case=r['case'], logT=r['logT_per_point'], seed=r['seed'], nucleation=r['nucleation_beta'],
                            beta_c=P['beta_c_TI_BASED'], E3={k: v[-1] for k, v in E.items()},
                            E_bc01={k: v[5] for k, v in E.items()}, first_cert_S1grid=fc))
    out = dict(per_pair=per_pair, runs=per_run)
    gains = [p['gain_coset_bc01'] for p in per_pair.values()]
    out['median_gain_coset_bc01'] = med(gains)
    out['median_gain_ball_bc01'] = med([p['gain_ball_bc01'] for p in per_pair.values()])
    out['median_gain_coset_3'] = med([p['gain_coset_3'] for p in per_pair.values()])
    out['range_gain_coset_bc01'] = [float(min(gains)), float(max(gains))] if gains else None
    budgets = {}
    for T in sorted(set(r['logT'] for r in per_run), reverse=True):
        R = [r for r in per_run if r['logT'] == T]
        cases = sorted(set(r['case'] for r in R))
        cnt = {}
        for k in ('point', 'ball', 'coset', 'point_coset', 'best'):
            cnt[k] = dict(pairs_all_seeds_at3=sum(all(r['E3'][k] > 0 for r in R if r['case'] == c) for c in cases),
                          pairs_all_seeds_at_bc01=sum(all(r['E_bc01'][k] > 0 for r in R if r['case'] == c) for c in cases),
                          runs_at3=sum(r['E3'][k] > 0 for r in R), runs_at_bc01=sum(r['E_bc01'][k] > 0 for r in R),
                          runs_any_S1grid=sum(r['first_cert_S1grid'][k] is not None for r in R))
        stuck = [r for r in R if r['first_cert_S1grid']['point'] is not None and
                 (r['nucleation'] is None or r['first_cert_S1grid']['point'] < r['nucleation'])]
        red = []
        for r in stuck:
            dp = r['first_cert_S1grid']['point'] - r['beta_c']
            dc = r['first_cert_S1grid']['point_coset'] - r['beta_c']
            red.append(1 - dc / dp if dp > 1e-9 else 0.0)
        budgets[f'T{T}'] = dict(n_pairs=len(cases), n_runs=len(R), certified=cnt, still_stuck_runs=len(stuck),
                                median_delay_reduction=med(red), delay_reductions=red)
    out['budgets'] = budgets
    b21 = budgets.get('T21')
    if b21:
        a = out['median_gain_coset_bc01'] is not None and out['median_gain_coset_bc01'] >= 3
        b1 = b21['certified']['point_coset']['pairs_all_seeds_at3'] - b21['certified']['point']['pairs_all_seeds_at3']
        b2 = b21['median_delay_reduction']
        out['decision'] = dict(a_median_gain_ge3=bool(a), b1_additional_pairs_at3=int(b1), b1_ok=bool(b1 >= 2),
                               b2_median_delay_reduction=b2, b2_ok=bool(b2 is not None and b2 >= 0.30),
                               main_text=bool(a and (b1 >= 2 or (b2 is not None and b2 >= 0.30))))
    return out


def silent():
    recs = {os.path.basename(f)[7:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(MF, 'silent_*.json')))}
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(FP, 'silent_*.json')))]
    res = dict(instances={}, counts={})
    rows = []
    for r in runs:
        t = r['info']['tag']
        if t not in recs:
            continue
        M = recs[t]
        for j, e in enumerate(r['RIGOROUS']['eval']):
            L = e['Lambda_FP']
            d = dict(tag=t, logT=r['logT_per_point'], seed=r['seed'], j=j, success=M['witness_success'])
            for src in ('graph_witness', 'planted'):
                x = M[src]['rows'][j]
                assert abs(x['beta'] - e['beta']) < 1e-6
                for k in ('point', 'ball', 'coset'):
                    d[f'{src}_{k}'] = x[k] - L
            rows.append(d)
    offs = ['-0.1', '+0.03', '+0.08', '+0.15']
    for T in sorted(set(d['logT'] for d in rows), reverse=True):
        for grp, sel in (('witness_success', True), ('witness_failure', False)):
            R = [d for d in rows if d['logT'] == T and d['success'] == sel]
            res['counts'][f'T{T}_{grp}'] = {f'{src}_{k}': [sum(d[f'{src}_{k}'] > 0 for d in R if d['j'] == j) for j in range(4)]
                                            for src in ('graph_witness', 'planted') for k in ('point', 'ball', 'coset')}
            res['counts'][f'T{T}_{grp}']['runs_per_offset'] = sum(d['j'] == 0 for d in R)
    res['offsets'] = offs
    res['gains'] = {t: dict(graph=[x['coset'] - x['point'] for x in M['graph_witness']['rows']],
                            planted=[x['coset'] - x['point'] for x in M['planted']['rows']],
                            S_graph=M['graph_witness']['S'], S_planted=M['planted']['S']) for t, M in recs.items()}
    # per-chain lambda = 0 test with the coset floor (graph witness), on the 'wrong' chains of the silent table
    lam = dict(chains=0, cert_point=0, cert_coset=0, min_margin_point=None, min_margin_coset=None)
    mp, mc = [], []
    for t, M in recs.items():
        sg = json.load(open(os.path.join(V3, 'results', 'silent_graph', t + '.json')))
        n = sg['n']
        lnf = float(gammaln(n + 1))
        for j, row in enumerate(sg['rows']):
            x = M['graph_witness']['rows'][j]
            for gr in row['groups']:
                for c in gr['chains']:
                    if c['label'] != 'wrong':
                        continue
                    lam['chains'] += 1
                    b = row['beta']
                    m_point = lnf + b * c['maxS'] - x['point'] - LN01      # certified iff < 0
                    m_coset = lnf + b * c['maxS'] - max(x['coset'], x['point']) - LN01
                    mp.append(m_point); mc.append(m_coset)
                    lam['cert_point'] += int(m_point < 0)
                    lam['cert_coset'] += int(m_coset < 0)
    if mp:
        lam['min_margin_point'] = float(min(mp)); lam['min_margin_coset'] = float(min(mc))
        lam['median_margin_coset'] = float(np.median(mc))
    res['per_chain_lambda0'] = lam
    return res


def table(p):
    L = [r'% R15 S1 draft (r15_mass_floor_analyze.py). All floors RIGOROUS; beta_c is the TI-based two-phase estimate.',
         r'\begin{tabular}{lccccc}', r'\toprule',
         r'TI budget & runs & \multicolumn{4}{c}{pairs certified at $\beta{=}3$, all seeds (runs certified at $\hat\beta_c{+}0.1$)} \\',
         r' & & point & ball & coset ELBO & best \\', r'\midrule']
    for k, v in p['budgets'].items():
        c = v['certified']
        L.append(f"$2^{{{k[1:]}}}$ & {v['n_runs']} & " + ' & '.join(
            f"{c[x]['pairs_all_seeds_at3']}/{v['n_pairs']} ({c[x]['runs_at_bc01']}/{v['n_runs']})" for x in ('point', 'ball', 'coset', 'best')) + r' \\')
    L += [r'\midrule',
          f"\\multicolumn{{6}}{{l}}{{median gain over the point floor at $\\hat\\beta_c{{+}}0.1$: coset {p['median_gain_coset_bc01']:.1f} nats, "
          f"ball {p['median_gain_ball_bc01']:.1f} nats (22 pairs)}} \\\\",
          r'\bottomrule', r'\end{tabular}']
    return '\n'.join(L) + '\n'


def main():
    p = protein()
    s = silent()
    out = dict(note='all floors RIGOROUS; beta_c, s_C TI-BASED (placement/context only)', protein=p, silent=s)
    json.dump(out, open(os.path.join(RES, 'summary_mass_floor.json'), 'w'), cls=Enc, indent=1)
    if p['per_pair'] and p['budgets']:
        open(os.path.join(V3, 'paper', 'tables', 'r15_mass_floor.tex'), 'w').write(table(p))
    return out


if __name__ == '__main__':
    o = main()
    p = o['protein']
    print(json.dumps(dict(median_gain_coset_bc01=p['median_gain_coset_bc01'], range=p['range_gain_coset_bc01'],
                          median_gain_ball_bc01=p['median_gain_ball_bc01'], decision=p.get('decision'),
                          budgets={k: {kk: vv for kk, vv in v.items() if kk != 'delay_reductions'} for k, v in p['budgets'].items()},
                          silent_counts=o['silent']['counts'], lam0=o['silent']['per_chain_lambda0']), cls=Enc, indent=1))
