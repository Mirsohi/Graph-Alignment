"""Generate LaTeX tables for v2/paper/tables/ from result JSONs."""
import json, os, glob, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
R = os.path.join(V2, 'results')
T = os.path.join(V2, 'paper', 'tables')
os.makedirs(T, exist_ok=True)


def nm(c):
    if c.startswith('syn'):
        return 'syn.\\ ER'
    return c.replace('pscdb_', '').replace('_', '.').replace('adk.open.closed', 'AdK o/c').replace('adk.crystal.control', 'AdK cryst.')


# ---------------- GT table
g = json.load(open(os.path.join(R, 'gt_comparison.json')))
order = ['pscdb_N_74', 'pscdb_CL_64', 'pscdb_O_19', 'syn_n40_d6_q0.1_s1', 'pscdb_N_141']
meths = [('twophase_single_run', '2P'), ('mh_random', 'MH-r'), ('mh_witness', 'MH-w'), ('annealed', 'Ann'),
         ('pt_tuned', 'PT'), ('wl_score', 'WL-$S$'), ('umb_k', 'Umb-$k$')]
cases = [c for c in order if c in g]
lines = ['\\begin{tabular}{l' + 'c' * len(cases) + '}', '\\toprule',
         ' & ' + ' & '.join(f"{nm(c)} ({g[c]['n']})" for c in cases) + ' \\\\', '\\midrule']
for m, lab in meths:
    cells = []
    for c in cases:
        s = g[c]['summary'].get(m)
        cells.append('--' if s is None else f"{s['max_dw']:.2f}/{s['max_tv']:.2f}")
    lines.append(lab + ' & ' + ' & '.join(cells) + ' \\\\')
lines.append('\\midrule')
lines.append('reference floor (TV) & ' + ' & '.join(f"{g[c]['summary']['gt_floor_max']:.2f}" for c in cases) + ' \\\\')
lines.append('reference SE of $\\wC$ & ' + ' & '.join(('--' if g[c]['source'] == 'mh' else f"{g[c]['summary'].get('gt_w_se_max', 0):.3f}") for c in cases) + ' \\\\')
lines.append('$\\#\\beta$ values & ' + ' & '.join(f"{g[c]['summary'].get('n_beta', 0)}" for c in cases) + ' \\\\')
lines.append('local budget/$\\beta$ & ' + ' & '.join(f"$2^{{{g[c]['summary'].get('budget_local_per_beta_log2')}}}$" for c in cases) + ' \\\\')
lines += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(T, 'gt.tex'), 'w').write('\n'.join(lines) + '\n')

# ---------------- BVS table (round 5: certified vs estimated FP-TI error kept separate)
rows = {}
for f in glob.glob(os.path.join(R, 'bvs', '*.json')):
    d = json.load(open(f))
    fam = re.sub(r'_s\d+$', '', d['name'])
    rows.setdefault(fam, []).append(d)
lines = ['\\begin{tabular}{lccccccc}', '\\toprule',
         'design & certified & true/estimated & 2P error & FP $<$ floor & stuck cert. & oracle cert. & $\\widehat R$ pass \\\\',
         ' & FP error & FP error & (exact) & & & & (stuck) \\\\', '\\midrule']
bvs_summary = {}
for fam, lab in (('exact20_strong', 'suppr., $p{=}20$'), ('p200', 'suppr., $p{=}200$'), ('decoy20', 'decoy, $p{=}20$'),
                 ('decoy200', 'decoy, $p{=}200$'), ('exact20_weak', 'weak, $p{=}20$')):
    if fam not in rows:
        continue
    ds = rows[fam]
    cert, esterr, tp_err, halfspread, below = [], [], [], [], 0
    stuck_c = stuck_n = orc_c = orc_n = rh_pass = rh_n = 0
    for d in ds:
        e = d['evidence']
        cert.append(e['witness_lower_bound'] - e['fp_ti'])
        if 'exact' in d:
            esterr.append(d['exact']['log_evidence'] - e['fp_ti'])
            tp_err.append(abs(e['two_phase'] - d['exact']['log_evidence']))
        else:
            esterr.append(e['two_phase'] - e['fp_ti'])
            reps = [np.logaddexp(c[-1], dd[-1]) for c, dd in zip(d['ti']['FC_reps'], d['ti']['FD_reps'])]
            halfspread.append(0.5 * (max(reps) - min(reps)))
        below += int(e['fp_certified_too_low'])
        for sname, c in d['chains'].items():
            for r in c['rows']:
                stuck = r['truth_incl'] < 0.5 and 'weak' not in fam
                if sname == 'oracle':
                    orc_n += 1
                    orc_c += int(r['crude'] < np.log(0.01))
                elif stuck:
                    stuck_n += 1
                    stuck_c += int(r['crude'] < np.log(0.01))
            if sname != 'oracle' and all(r['truth_incl'] < 0.5 for r in c['rows']) and 'weak' not in fam:
                rh_n += 1
                rh_pass += int(c['rhat_ell'] < 1.01 and c['rhat_size'] < 1.01)
    ce = f"$[{min(cert):.1f},{max(cert):.1f}]$"
    ee = f"$[{min(esterr):.1f},{max(esterr):.1f}]$" + ("$^\\dagger$" if halfspread else '')
    tpe = f"$\\le${max(tp_err):.3f}" if tp_err else '--'
    bvs_summary[fam] = dict(certified=[min(cert), max(cert)], estimated=[min(esterr), max(esterr)],
                            twophase_halfspread_max=max(halfspread) if halfspread else None,
                            exact=bool(tp_err), below=below, n=len(ds))
    lines.append(f"{lab} & {ce} & {ee} & {tpe} & {below}/{len(ds)} & {stuck_c}/{stuck_n} & {orc_c}/{orc_n} & {rh_pass}/{rh_n} \\\\")
lines += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(T, 'bvs.tex'), 'w').write('\n'.join(lines) + '\n')
json.dump(bvs_summary, open(os.path.join(R, 'summary_bvs_errors.json'), 'w'), indent=1)

# ---------------- validity table (appendix)
val = {os.path.basename(f)[:-8]: json.load(open(f)) for f in glob.glob(os.path.join(R, 'validity', '*_s1.json'))}
v22 = {os.path.basename(f)[:-8]: json.load(open(f)) for f in glob.glob(os.path.join(R, 'validity22', '*_s2.json'))}
sv = json.load(open(os.path.join(R, 'summary_v1chains.json')))['status']
lines = ['\\begin{tabular}{lrrrrrrrl}', '\\toprule',
         'pair & $n$ & $\\hat\\beta_c$ & edge & melt $\\Delta S$ & $\\mu_{\\rm win}(C)$ ($2^{21}$) & ($2^{22}$) & basins $(k,\\text{mass})$ & survey label (historical) \\\\',
         '\\midrule']
for c in sorted(val, key=lambda c: val[c]['beta_c']):
    v = val[c]
    e = max(v['edge']['edge_D'] + v['edge']['edge_C'])
    m = v['melt']
    dS = abs(m['partial_start']['meanS'] - m['random_start']['meanS'])
    bas = ', '.join(f"({b['k_peak']},{b['mass']:.2f})" for b in v['basins'])
    w22 = f"{v22[c]['wC_window']:.2f}" if c in v22 else '--'
    st = sv.get(c, '?').replace('window-inconclusive', 'unresolved (W)').replace('C-metastable', 'unresolved (2C)').replace(' (2nd window pass)', '$^\\ast$')
    if c in ('pscdb_B_44', 'pscdb_N_74', 'pscdb_N_217'):
        st += '$^\\dagger$'   # fails one rule on the fresh-seed repeat (round 6/7)
    lines.append(f"{nm(c)} & {v['n']} & {v['beta_c']:.3f} & {e:.3f} & {dS:.1f} & {v['wC_window']:.2f} & {w22} & {bas} & {st} \\\\")
lines += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(T, 'valid.tex'), 'w').write('\n'.join(lines) + '\n')

# ---------------- exact checks
ex = json.load(open(os.path.join(R, 'exact_twophase.json')))
lines = ['\\begin{tabular}{lrrrrrr}', '\\toprule',
         'instance & err $\\log Z$ & err $F_C$ & err $\\omega_{\\mathrm C}$ & $\\beta_c$ exact & $\\beta_c$ est. & $\\mu_6(W\\mid C)$ \\\\', '\\midrule']
for k, v in ex.items():
    lines.append(f"{k.replace('_', ' ')} & {v['maxerr_logZ']:.3f} & {v['maxerr_FC']:.3f} & {v['maxerr_wcond']:.4f} & "
                 f"{v['beta_c_exact']:.3f} & {v['beta_c_est']:.3f} & {v['fracW_hi']:.3f} \\\\")
lines += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(T, 'exact.tex'), 'w').write('\n'.join(lines) + '\n')
ne = json.load(open(os.path.join(R, 'exact_nested.json')))
lines = ['\\begin{tabular}{lrrrrr}', '\\toprule',
         'instance, $\\beta$ & exact $\\log\\mu_\\beta(W\\mid C)$ & estimate (mean) & sd (5 runs) & reported SE & fractions \\\\', '\\midrule']
for r in ne:
    lines.append(f"n{r['n']} s{r['seed']}, {r['beta']} & ${r['exact']:.3f}$ & ${r['est_mean']:.3f}$ & {r['est_sd']:.3f} & "
                 f"{r['mean_reported_se']:.3f} & {len(r['levels']) - 1} \\\\")
lines += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(T, 'nested.tex'), 'w').write('\n'.join(lines) + '\n')

# ---------------- alarm table
al = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, 'alarm', '*.json')))]
if al:
    diag = [('rhat_alarm', '$\\widehat R\\ge1.01$', 'g'), ('rstar_alarm', '$R^\\ast\\ge1.05$', 'g'), ('d_alarm', '$\\Delta/\\Delta_{\\rm iid}\\ge1.5$', 'g'),
            ('crude_cert', 'rigorous certificate ($\\lambda{=}0$)', 'c'), ('tilted_cert', 'TI-based tilted', 'c'),
            ('twophase_tv_alarm', 'TI-based 2P occupancy error$\\ge$0.5', 'c')]
    cnt = {d[0]: dict(fa=0, nfa=0, det=0, ndet=0) for d in diag}
    for a in al:
        for gr in a['groups']:
            labs = [c['label'] for c in gr['chains']]
            for key, lab, lev in diag:
                if lev == 'g':
                    # R-hat = NaN (identical constant chains) counts as no alarm
                    al_ = bool(gr['rhat'] >= 1.01) if key == 'rhat_alarm' else bool(gr[key])
                    if gr['start'] == 'planted':   # exactly stationary controls only
                        cnt[key]['nfa'] += 1
                        cnt[key]['fa'] += int(al_)
                    elif any(l == 'wrong' for l in labs):
                        cnt[key]['ndet'] += 1
                        cnt[key]['det'] += int(al_)
                else:
                    for c in gr['chains']:
                        if gr['start'] == 'planted':   # exactly stationary controls only
                            cnt[key]['nfa'] += 1
                            cnt[key]['fa'] += int(c[key])
                        elif c['label'] == 'wrong':
                            cnt[key]['ndet'] += 1
                            cnt[key]['det'] += int(c[key])
    lines = ['\\begin{tabular}{lcc}', '\\toprule', 'diagnostic & false alarms (planted starts) & detections \\\\', '\\midrule']
    for key, lab, lev in diag:
        c = cnt[key]
        lines.append(f"{lab} ({'groups' if lev == 'g' else 'chains'}) & {c['fa']}/{c['nfa']} & {c['det']}/{c['ndet']} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(T, 'alarm.tex'), 'w').write('\n'.join(lines) + '\n')
    json.dump(cnt, open(os.path.join(R, 'summary_alarm.json'), 'w'), indent=1)
    print('alarm', cnt)
print('tables written')

# ---------------- silent-regime table: see make_silent.py (round 3)
import runpy
runpy.run_path(os.path.join(HERE, 'make_silent.py'), run_name='__main__')
runpy.run_path(os.path.join(HERE, 'make_heldout_table.py'), run_name='__main__')

# ---------------- released procedure on all 22 pairs (round 10; exhaustive status table)
AB = dict(edge='E', D_two_start='2D', C_two_start='2C', window='W', middle_basin='M')
fr = lambda rules: ','.join(AB[k] for k, v in rules.items() if not v) or 'ok'
vmr = {}
for f in glob.glob(os.path.join(R, 'validity_multi', '*.json')):
    d = json.load(open(f))
    if 'rows' in d:
        vmr[d['case']] = [r for r in d['rows'] if r['offset'] == 0.0][0]
mrc = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(R, 'middle_recheck', '*.json'))}
sre = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(R, 'survey_retry', '*.json'))}
SURVEY_FAIL = {'window-inconclusive': 'W', 'C-metastable': '2C'}
lines = ['\\begin{tabular}{lrrllll}', '\\toprule',
         'pair & $n$ & $\\hat\\beta_c$ & survey label & first attempt (seed a / b) & retry (seed c / d) & released status \\\\', '\\midrule']
counts = {}
rel = {}
for c in sorted(val, key=lambda c: val[c]['beta_c']):
    v = val[c]
    lab = sv.get(c, '?')
    if c in mrc:                                   # survey multi-phase: two fresh seeds
        first = mrc[c]['status']
        fa = ' / '.join(fr(r['rules']) for r in mrc[c]['runs']) + '$^\\dagger$'
    elif lab.startswith('pass'):                   # survey seed + fresh-seed repeat
        rr = vmr[c]['rules']
        first = 'pass' if all(rr.values()) else 'unresolved'
        fa = 'ok / ' + fr(rr)
    else:                                          # survey seed failed a rule, no middle basin
        n_ = v['n']
        mid = any(b['mass'] > 0.05 and 0.1 * n_ < b['k_peak'] < v['k0'] + 0.05 * n_ for b in v['basins'])
        assert not mid, c
        first = 'unresolved'
        fa = SURVEY_FAIL.get(lab, lab) + ' / --'
    if first == 'unresolved' and c in sre:
        ret = ' / '.join(fr(r['rules']) for r in sre[c]['runs'])
        final = sre[c]['final_status']
    else:
        ret, final = '--', first
    rel[c] = dict(survey=lab, first=first, final=final)
    counts[final] = counts.get(final, 0) + 1
    labt = lab.replace('window-inconclusive', 'unresolved (W)').replace('C-metastable', 'unresolved (2C)').replace(' (2nd window pass)', '$^\\ast$')
    lines.append(f"{nm(c)} & {v['n']} & {v['beta_c']:.3f} & {labt} & {fa} & {ret} & {final} \\\\")
lines += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(T, 'released.tex'), 'w').write('\n'.join(lines) + '\n')
json.dump(dict(pairs=rel, counts=counts), open(os.path.join(R, 'summary_released_22.json'), 'w'), indent=1)
print('released status counts', counts)
