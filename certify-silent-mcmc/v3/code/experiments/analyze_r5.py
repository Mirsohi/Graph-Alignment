"""Round-5 analysis: end-to-end pipeline replicates (error budget, spread), full cost
accounting, and the matched-total-cost comparators of baselines3.py, all against the
GT references of gt_comparison.json.  Writes results/summary_r5.json and
paper/tables/{budget,cost,pt}.tex."""
import json, os, glob, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
R = os.path.join(V2, 'results')
T = os.path.join(V2, 'paper', 'tables')
sys.path.insert(0, HERE)
from analyze_gt import load_gt, rowtv

CASES = ['pscdb_N_74', 'pscdb_CL_64', 'pscdb_O_19', 'syn_n40_d6_q0.1_s1', 'pscdb_N_141']
SRC = {'pscdb_N_74': 'mh', 'pscdb_CL_64': 'umb', 'pscdb_O_19': 'umb', 'syn_n40_d6_q0.1_s1': 'umb', 'pscdb_N_141': 'win4'}
NAME = {'pscdb_N_74': 'N.74', 'pscdb_CL_64': 'CL.64', 'pscdb_O_19': 'O.19', 'syn_n40_d6_q0.1_s1': 'syn.', 'pscdb_N_141': 'N.141'}
gtc = json.load(open(os.path.join(R, 'gt_comparison.json')))
out = dict(budget={}, cost={}, comparators={})
lg2 = lambda x: float(np.log2(x))

for c in CASES:
    reps = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, 'pipeline_reps', f'{c}_s*.json')))]
    if not reps:
        continue
    g = gtc[c]
    n = g['n']
    gtb = [r['beta'] for r in g['rows']]
    wgt = {r['beta']: r['w_gt'] for r in g['rows']}
    bcg = g['beta_c_gt']
    # beta in the GT grid nearest the GT transition
    bstar = min(gtb, key=lambda b: abs(b - bcg))
    per = []
    for rp in reps:
        rows = {round(r['beta'], 6): r for r in rp['rows'][:len(gtb)]}
        rs = rows[round(bstar, 6)]
        per.append(dict(beta_c=rp['beta_c'], maxdw=max(abs(rows[round(b, 6)]['wC'] - wgt[b]) for b in gtb),
                        w_star=rs['wC'], logit_star=rs['logit'], logZ_star=rs['logZ'],
                        se_mc=rs['se_logit_mc'], se_ref=rs['se_logit_ref'], disc=rs['disc_logit_half'],
                        se_logZ=float(np.hypot(rs['se_logZ_mc'], rs['se_logZ_ref'])), disc_logZ=rs['disc_logZ_half'],
                        latent=rp['latent_heat'], cost=rp['cost'], witness_ok=rp['Sw'] >= rp['Sw_cached'],
                        agree=rp['witness_agree_true']))
    bcs = np.array([p['beta_c'] for p in per])
    lat = float(np.mean([p['latent'] for p in per]))
    se_stat = float(np.mean([np.hypot(p['se_mc'], p['se_ref']) for p in per]))
    disc = float(np.mean([p['disc'] for p in per]))
    ws = np.array([p['w_star'] for p in per])
    out['budget'][c] = dict(
        n=n, n_reps=len(per), beta_star=bstar, w_gt_star=wgt[bstar], beta_c_gt=bcg,
        beta_c_reps=bcs.tolist(), beta_c_mean=float(bcs.mean()), beta_c_sd=float(bcs.std(ddof=1)),
        maxdw_reps=[p['maxdw'] for p in per], w_star_reps=ws.tolist(), w_star_sd=float(ws.std(ddof=1)),
        logit_sd_endtoend=float(np.std([p['logit_star'] for p in per], ddof=1)),
        se_logit_mc=float(np.mean([p['se_mc'] for p in per])), se_logit_ref=float(np.mean([p['se_ref'] for p in per])),
        disc_logit_half_estmeans=disc,
        beta_c_se_stat=se_stat / lat, beta_c_disc_half=disc / lat, latent_heat=lat,
        logZ_sd_endtoend=float(np.std([p['logZ_star'] for p in per], ddof=1)),
        se_logZ_stat=float(np.mean([p['se_logZ'] for p in per])), disc_logZ_half=float(np.mean([p['disc_logZ'] for p in per])),
        witness_ok=[p['witness_ok'] for p in per])
    # cost accounting (proposals)
    wit = float(np.mean([p['cost']['witness'] for p in per]))
    est = float(np.mean([p['cost']['estimator'] for p in per]))
    nes = float(np.mean([p['cost']['nested'] for p in per]))
    valid = (n // 2 + 2) * 2 ** 21 + 2 * 2 ** 22 + 2 * 2 ** 23
    tot = wit + est + nes + valid
    out['cost'][c] = dict(witness=wit, estimator=est, nested=nes, validation=valid, total=tot,
                          restarts=[p['cost']['refinement_restarts'] for p in per],
                          log2=dict(witness=lg2(wit), estimator=lg2(est), nested=lg2(nes), validation=lg2(valid),
                                    total=lg2(tot)))

# matched-total-cost comparators (seed 0 = round 5; seeds 1, 2 = round-6 replication)
for c in CASES:
    files = sorted(glob.glob(os.path.join(R, 'baselines3', c + '*.json')))
    if not files:
        continue
    gt = load_gt(c, SRC[c])
    res = {}
    for fpath in files:
        b3 = json.load(open(fpath))
        M = np.load(fpath[:-5] + '_marg.npz')
        for m, v in b3['methods'].items():
            dws, tvs = [], []
            for bb in b3['betas']:
                dws.append(abs(v['w'][str(bb)] - gt[float(bb)]['w']))
                key = f'{bb}_{m}'
                if key in M.files:
                    tvs.append(rowtv(M[key], gt[float(bb)]['P']))
            e = res.setdefault(m, dict(max_dw=[], max_tv=[], sec=[], seeds=[]))
            e['max_dw'].append(float(max(dws)))
            e['max_tv'].append(float(max(tvs)) if tvs else None)
            e['sec'].append(float(v.get('sec', np.nan)))
            e['seeds'].append(b3.get('seed', 0))
            if m == 'wl_ext':
                e.setdefault('max_visited_S', []).append(v['max_visited_S'])
                e['Sw'] = v['Sw']
    res['logB'] = 29
    res['twophase_single_run'] = gtc[c]['summary']['twophase_single_run']
    if c in out['budget']:
        reps = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, 'pipeline_reps', f'{c}_s*.json')))]
        vm = os.path.join(R, 'validity_multi', c + '.json')
        vsec = [r['sec'] for r in json.load(open(vm))['rows'] if r['offset'] == 0.0][0] if os.path.exists(vm) else None
        res['twophase_pipeline'] = dict(max_dw=out['budget'][c]['maxdw_reps'], sec=[r['sec'] for r in reps], validity_sec=vsec)
    out['comparators'][c] = res
# mesh halving (round 6)
mh = {}
for c in ('pscdb_O_19', 'pscdb_N_141'):
    hs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, 'mesh_half', f'{c}_h001_s*.json')))]
    if not hs:
        continue
    reps = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, 'pipeline_reps', f'{c}_s*.json')))]
    bstar = hs[0]['beta_star']
    std = dict(beta_c=[r['beta_c'] for r in reps],
               logit=[[x for x in r['rows'] if abs(x['beta'] - bstar) < 1e-9][0]['logit'] for r in reps],
               logZ=[[x for x in r['rows'] if abs(x['beta'] - bstar) < 1e-9][0]['logZ'] for r in reps])
    half = dict(beta_c=[h['beta_c'] for h in hs], logit=[h['logit_star'] for h in hs], logZ=[h['logZ_star'] for h in hs])
    d = {}
    for k in ('beta_c', 'logit', 'logZ'):
        a_, b_ = np.array(std[k]), np.array(half[k])
        d[k] = dict(std_mean=float(a_.mean()), std_sd=float(a_.std(ddof=1)), n_std=len(a_), half_mean=float(b_.mean()),
                    half_sd=float(b_.std(ddof=1)), n_half=len(b_), diff=float(b_.mean() - a_.mean()),
                    se_diff=float(np.sqrt(a_.var(ddof=1) / len(a_) + b_.var(ddof=1) / len(b_))))
    mh[c] = dict(beta_star=bstar, **d)
fb = os.path.join(R, 'mesh_half', 'bvs_p200_4rep.json')
if os.path.exists(fb):
    b = json.load(open(fb))
    a_, b_ = np.array(b['logZ_2P']['40']), np.array(b['logZ_2P']['80'])
    mh['bvs_p200'] = dict(logZ=dict(std_mean=float(a_.mean()), std_sd=float(a_.std(ddof=1)), n_std=len(a_),
                                    half_mean=float(b_.mean()), half_sd=float(b_.std(ddof=1)), n_half=len(b_),
                                    diff=float(b_.mean() - a_.mean()),
                                    se_diff=float(np.sqrt(a_.var(ddof=1) / len(a_) + b_.var(ddof=1) / len(b_)))))
out['mesh_half'] = mh
if mh:
    L = ['\\begin{tabular}{llccc}', '\\toprule', 'instance & quantity & standard mesh & halved mesh & difference \\\\', '\\midrule']
    for c, lab in (('pscdb_O_19', 'O.19'), ('pscdb_N_141', 'N.141'), ('bvs_p200', 'VS, $p{=}200$')):
        if c not in mh:
            continue
        for k, kl in (('beta_c', '$\\hat\\beta_c$'), ('logit', 'logit $\\hat\\omega_{\\mathrm C}(\\beta^\\ast)$'), ('logZ', '$\\log\\hat Z$')):
            if k not in mh[c]:
                continue
            e = mh[c][k]
            f = '.4f' if k == 'beta_c' else '.3f'
            L.append(f"{lab} & {kl} & ${e['std_mean']:{f}}\\pm{e['std_sd']:{f}}$ ({e['n_std']}) & ${e['half_mean']:{f}}\\pm{e['half_sd']:{f}}$ ({e['n_half']}) & ${e['diff']:+{f}}\\pm{e['se_diff']:{f}}$ \\\\")
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(T, 'mesh.tex'), 'w').write('\n'.join(L) + '\n')
# validity at several temperatures (round 6)
vm = {}
for f in sorted(glob.glob(os.path.join(R, 'validity_multi', 'p*.json')) + glob.glob(os.path.join(R, 'validity_multi', 's*.json'))):
    v = json.load(open(f))
    d = json.load(open(os.path.join(R, 'twophase', v['source'])))
    gD, gC = np.array(d['gridD']), np.array(d['gridC'])
    rows = []
    for r in v['rows']:
        iD, iC = int(np.argmin(np.abs(gD - r['beta']))), int(np.argmin(np.abs(gC - r['beta'])))
        eD = max(x['edge_D'][iD] for x in d['reps'])
        eC = max(x['edge_C'][iC] for x in d['reps'])
        wE = r['w_est']
        # region rules apply to regions carrying at least 1% of the estimated mass
        needD, needC = (1 - wE) >= 0.01, wE >= 0.01
        ru = r['rules']
        weighted = dict(edge=(eD < 0.01 or not needD) and (eC < 0.01 or not needC),
                        D_two_start=ru['D_two_start'] or not needD, C_two_start=ru['C_two_start'] or not needC,
                        window=ru['window'], middle_basin=ru['middle_basin'])
        lowfail = bool(((not needD) and (eD >= 0.01 or not ru['D_two_start'])) or ((not needC) and (eC >= 0.01 or not ru['C_two_start'])))
        rows.append(dict(offset=r['offset'], beta=r['beta'], w_est=wE, w_window=r['w_window'], edge_D=eD, edge_C=eC, lowregion_fail=lowfail,
                         raw=ru, raw_pass=r['passed'], weighted=weighted, weighted_pass=bool(all(weighted.values()))))
    vm[v['case']] = dict(beta_c=v['beta_c'], rows=rows)
rc = os.path.join(R, 'validity_multi', 'window_recheck.json')
out['validity_multi'] = dict(cases=vm, window_recheck=json.load(open(rc)) if os.path.exists(rc) else None)
if vm:
    L = ['\\begin{tabular}{l' + 'c' * 5 + '}', '\\toprule', 'instance & $\\hat\\beta_c-0.25$ & $-0.1$ & $\\hat\\beta_c$ & $+0.1$ & $+0.25$ \\\\', '\\midrule']
    AB = dict(edge='E', D_two_start='2D', C_two_start='2C', window='W', middle_basin='M')
    sym = lambda r: ('pass' if r['weighted_pass'] else 'fail (' + ','.join(AB[k] for k, x in r['weighted'].items() if not x) + ')') + ('$^\\ast$' if r['lowregion_fail'] else '')
    for c in vm:
        L.append(NAME.get(c, c.replace('pscdb_', '').replace('_', '.')) + ' & ' + ' & '.join(sym(r) for r in vm[c]['rows']) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(T, 'validity_multi.tex'), 'w').write('\n'.join(L) + '\n')
# per-run status of every full-pipeline replicate (round 7)
runs = []
for c in CASES:
    g = gtc[c]
    wgt = {r['beta']: r['w_gt'] for r in g['rows']}
    for f in sorted(glob.glob(os.path.join(R, 'pipeline_reps', f'{c}_s*.json'))):
        rp = json.load(open(f))
        st = os.path.join(R, 'replicate_status', os.path.basename(f))
        sd = json.load(open(st)) if os.path.exists(st) else None
        rt = os.path.join(R, 'retry_status', os.path.basename(f))   # round 10: bounded retry
        rd = json.load(open(rt)) if os.path.exists(rt) else None
        rows = {round(r['beta'], 6): r for r in rp['rows']}
        mdw = max(abs(rows[round(b, 6)]['wC'] - wgt[b]) for b in wgt)
        runs.append(dict(case=c, seed=rp['seed'], witness_best=bool(rp['Sw'] >= rp['Sw_cached']),
                         restarts=rp['cost']['refinement_restarts'], beta_c=rp['beta_c'], max_dw=mdw, included=True,
                         status=sd['status'] if sd else None, failed=sd['failed_rules'] if sd else None,
                         cost_pipeline=rp['cost_total'], cost_diag=sd['diag_proposals'] if sd else None,
                         sec_pipeline=rp['sec'], sec_diag=sd['diag_sec'] if sd else None,
                         retry_status=rd['retry_status'] if rd else None, retry_failed=rd['retry_failed_rules'] if rd else None,
                         final_status=(rd['final_status'] if rd else (sd['status'] if sd else None)),
                         cost_retry=rd['retry_proposals'] if rd else 0, sec_retry=rd['retry_sec'] if rd else None))
out['runs'] = runs
AB = dict(edge='E', D_two_start='2D', C_two_start='2C', window='W', middle_basin='M')
if runs:
    L = ['\\begin{tabular}{llccccccc}', '\\toprule',
         'inst. & run & restarts & $\\hat\\beta_c$ & max $|\\Delta\\wC|$ & first attempt (seed 1 / 2) & retry (seed 3 / 4) & proposals & seconds$^\\ddagger$ \\\\', '\\midrule']
    for r in runs:
        dg = ('--' if r['status'] is None else r['status'] + ' (' + ' / '.join((','.join(AB[x] for x in fr) or 'ok') for fr in r['failed']) + ')')
        rtr = ('--' if r['retry_status'] is None else r['retry_status'] + ' (' + ' / '.join((','.join(AB[x] for x in fr) or 'ok') for fr in r['retry_failed']) + ')')
        cost = f"$2^{{{np.log2(r['cost_pipeline']):.1f}}}$ + $2^{{{np.log2(r['cost_diag']):.1f}}}$" if r['cost_diag'] else f"$2^{{{np.log2(r['cost_pipeline']):.1f}}}$"
        if r['cost_retry']:
            cost += f" + $2^{{{np.log2(r['cost_retry']):.1f}}}$"
        sec = f"{r['sec_pipeline']:.0f} + {r['sec_diag']:.0f}" if r['sec_diag'] else f"{r['sec_pipeline']:.0f}"
        if r['sec_retry']:
            sec += f" + {r['sec_retry']:.0f}"
        L.append(f"{NAME[r['case']]} & {r['seed']} & {r['restarts']} & {r['beta_c']:.4f} & {r['max_dw']:.3f} & {dg} & {rtr} & {cost} & {sec} \\\\")
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(T, 'runs.tex'), 'w').write('\n'.join(L) + '\n')
json.dump(out, open(os.path.join(R, 'summary_r5.json'), 'w'), indent=1)

cs = [c for c in CASES if c in out['budget']]
# ---- error-budget table
L = ['\\begin{tabular}{l' + 'c' * len(cs) + '}', '\\toprule', ' & ' + ' & '.join(f"{NAME[c]} ({out['budget'][c]['n']})" for c in cs) + ' \\\\', '\\midrule']
row = lambda lab, f: L.append(lab + ' & ' + ' & '.join(f(out['budget'][c]) for c in cs) + ' \\\\')
row('$\\hat\\beta_c$ (mean$\\pm$sd over runs)', lambda d: f"{d['beta_c_mean']:.4f}$\\pm${d['beta_c_sd']:.4f}")
row('independent full-pipeline runs', lambda d: f"{d['n_reps']}")
row('$\\beta_c$ reference', lambda d: f"{d['beta_c_gt']:.4f}")
row('max $|\\hat\\omega_{\\mathrm C}-\\wC^{\\rm ref}|$ (range)', lambda d: f"{min(d['maxdw_reps']):.2f}--{max(d['maxdw_reps']):.2f}")
L.append('\\midrule')
row('logit $\\hat\\omega_{\\mathrm C}$ at $\\beta^\\ast$: MC SE', lambda d: f"{d['se_logit_mc']:.2f}")
row('\\quad anchor SE ($\\mu(W|C)$)', lambda d: f"{d['se_logit_ref']:.2f}")
row('\\quad Riemann half-width$^\\ast$', lambda d: f"{d['disc_logit_half_estmeans']:.2f}")
row('\\quad end-to-end sd', lambda d: f"{d['logit_sd_endtoend']:.2f}")
row('$\\log Z(\\beta^\\ast)$: SE / half-width$^\\ast$ / sd', lambda d: f"{d['se_logZ_stat']:.2f} / {d['disc_logZ_half']:.2f} / {d['logZ_sd_endtoend']:.2f}")
row('$\\beta_c$: SE / half-width$^\\ast$', lambda d: f"{d['beta_c_se_stat']:.4f} / {d['beta_c_disc_half']:.4f}")
L += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(T, 'budget.tex'), 'w').write('\n'.join(L) + '\n')
# ---- cost table
L = ['\\begin{tabular}{l' + 'c' * len(cs) + '}', '\\toprule', ' & ' + ' & '.join(NAME[c] for c in cs) + ' \\\\', '\\midrule']
rowc = lambda lab, k: L.append(lab + ' & ' + ' & '.join(f"$2^{{{out['cost'][c]['log2'][k]:.1f}}}$" for c in cs) + ' \\\\')
rowc('witness search (4 restarts)', 'witness')
rowc('estimator (incl.\\ refinement restarts)', 'estimator')
rowc('nested reference', 'nested')
rowc('validity checks', 'validation')
rowc('pipeline with diagnostic checks (one seed)', 'total')
L += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(T, 'cost.tex'), 'w').write('\n'.join(L) + '\n')
# ---- comparator table (median [min, max] of max |dw| over independent runs; wall-clock per run)
cc = [c for c in CASES if c in out['comparators']]
fmt = lambda xs: f"{np.median(xs):.2f} [{min(xs):.2f},{max(xs):.2f}]" if len(xs) > 1 else f"{xs[0]:.2f}"
if cc:
    L = ['\\begin{tabular}{l' + 'c' * len(cc) + '}', '\\toprule', ' & ' + ' & '.join(NAME[c] for c in cc) + ' \\\\', '\\midrule',
         '\\multicolumn{' + str(len(cc) + 1) + '}{l}{max $|\\hat\\omega_{\\mathrm C}-\\wC^{\\rm ref}|$: median [min, max] over independent runs} \\\\']
    L.append('2P, full pipeline & ' + ' & '.join(fmt(out['comparators'][c]['twophase_pipeline']['max_dw']) + f" ({len(out['comparators'][c]['twophase_pipeline']['max_dw'])})" for c in cc) + ' \\\\')
    for m, lab in (('pt_broad', 'PT, broad ladder'), ('pt_broad_w', 'PT, broad, witness-seeded'),
                   ('wl_ext', 'WL-$S$, range to $\\min(|E_A|,|E_B|)$'), ('win_direct', 'direct windows')):
        L.append(lab + ' & ' + ' & '.join((fmt(out['comparators'][c][m]['max_dw']) + f" ({len(out['comparators'][c][m]['max_dw'])})")
                                          if m in out['comparators'][c] else '--' for c in cc) + ' \\\\')
    L += ['\\midrule', '\\multicolumn{' + str(len(cc) + 1) + '}{l}{wall-clock seconds per run, one process; 2P with one diagnostic seed; other core busy} \\\\']
    L.append('2P: pipeline + one-seed checks & ' + ' & '.join(
        f"{np.mean(out['comparators'][c]['twophase_pipeline']['sec']):.0f} + {out['comparators'][c]['twophase_pipeline']['validity_sec']:.0f}"
        if out['comparators'][c]['twophase_pipeline']['validity_sec'] else '--' for c in cc) + ' \\\\')
    for m, lab in (('pt_broad', 'PT, broad'), ('pt_broad_w', 'PT, witness-seeded'), ('wl_ext', 'WL-$S$ ext.'), ('win_direct', 'direct windows')):
        L.append(lab + ' & ' + ' & '.join(f"{np.nanmean(out['comparators'][c][m]['sec']):.0f}" if m in out['comparators'][c] else '--' for c in cc) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(T, 'pt.tex'), 'w').write('\n'.join(L) + '\n')
print('ok')
