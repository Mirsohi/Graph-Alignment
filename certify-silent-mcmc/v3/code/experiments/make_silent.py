"""Silent-regime diagnostic table (round 3): exact BVS (results/silent_bvs) and
synthetic graph alignment with window-GT labels (results/silent_graph).
False-alarm populations:
  BVS   : groups/chains started from EXACT posterior draws (exactly stationary);
  graph : groups/chains labelled correct by the window GT (|nu* - w_GT| < 0.2).
R-hat is the rank-normalised folded split-R-hat of Vehtari et al. (2021)
(gacert.core.split_rhat; BVS: max over ell and model size).
Detection populations: chains labelled wrong (|nu* - w_GT| > 0.5); groups whose
labelled chains are all wrong.
Writes paper/tables/silent.tex and results/summary_silent.json."""
import json, os, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
R = os.path.join(V2, 'results')
T = os.path.join(V2, 'paper', 'tables')

DIAG = [('rhat', '$\\widehat R\\ge1.01$', 'g'), ('rhat105', '$\\widehat R\\ge1.05$', 'g'), ('rstar', '$R^\\ast\\ge1.05$', 'g'), ('dratio', '$\\Delta/\\Delta_{\\rm iid}\\ge1.5$', 'g'),
        ('crude_cert', 'rigorous cert., $\\lambda{=}0$', 'c'), ('tilted_cert', 'TI-based tilted', 'c'),
        ('twophase_tv_alarm', 'TI-based occ.\\ err.\\ ${}\\ge0.5$', 'c')]


def new():
    return {k: dict(fa=0, nfa=0, det=0, ndet=0) for k, _, _ in DIAG}


def galarm(gr, key):
    # R-hat is undefined (NaN) when every chain is frozen in one state (planted controls);
    # undefined values are counted as 'no alarm' and reported separately.
    if key == 'rhat':
        return bool(gr['rhat'] >= 1.01)
    if key == 'rhat105':
        return bool(gr['rhat'] >= 1.05)
    if key == 'rstar':
        return bool(gr['rstar'] >= 1.05)
    if key == 'dratio':
        return bool(gr['d_ratio'] >= 1.5) if 'd_ratio' in gr else None
    raise KeyError


def add_group(cnt, gr, fa_group, fa_chain_pred):
    labs = [c['label'] for c in gr['chains'] if c['label'] != 'ambiguous']
    wrong_group = bool(labs) and all(l == 'wrong' for l in labs)
    for key, _, lev in DIAG:
        if lev == 'g':
            a = galarm(gr, key)
            if a is None:
                continue
            if fa_group:
                cnt[key]['nfa'] += 1; cnt[key]['fa'] += int(a)
            elif wrong_group:
                cnt[key]['ndet'] += 1; cnt[key]['det'] += int(a)
        else:
            for c in gr['chains']:
                if fa_chain_pred(c):
                    cnt[key]['nfa'] += 1; cnt[key]['fa'] += int(c[key])
                elif c['label'] == 'wrong':
                    cnt[key]['ndet'] += 1; cnt[key]['det'] += int(c[key])


out = {}
# BVS (exact labels)
bv = new()
extra = dict(rhat_alarm_on_correct_nonexact_groups=0, correct_nonexact_groups=0)
for f in sorted(glob.glob(os.path.join(R, 'silent_bvs', '*.json'))):
    a = json.load(open(f))
    for gr in a['groups']:
        ex = gr['start'] == 'exact'
        add_group(bv, gr, ex, lambda c, ex=ex: ex)
        labs = [c['label'] for c in gr['chains']]
        if not ex and all(l == 'correct' for l in labs):
            extra['correct_nonexact_groups'] += 1
            extra['rhat_alarm_on_correct_nonexact_groups'] += int(gr['rhat'] >= 1.01)
out['bvs'] = bv
out['bvs_extra'] = extra
# graph alignment (window-GT labels), all evaluation betas, and the subset above beta_c
gg, ga = new(), new()
inst = []
for f in sorted(glob.glob(os.path.join(R, 'silent_graph', '*.json'))):
    a = json.load(open(f))
    if 'rows' not in a:
        inst.append(dict(tag=a['tag'], note=a.get('note')))
        continue
    inst.append(dict(tag=a['tag'], beta_c_GT=a['beta_c_GT'], beta_c_2P=a['beta_c_2P'],
                     witness_agree=a['witness_agree_planted']))
    for row in a['rows']:
        for gr in row['groups']:
            labs = [c['label'] for c in gr['chains'] if c['label'] != 'ambiguous']
            okg = bool(labs) and len(labs) == len(gr['chains']) and all(l == 'correct' for l in labs)
            add_group(gg, gr, okg, lambda c: c['label'] == 'correct')
            if row['dbeta'] > 0:
                add_group(ga, gr, okg, lambda c: c['label'] == 'correct')
# silent subset: random-start groups above beta_c whose chains are all wrong: how many pass R-hat
sil = dict(wrong_random_groups_above=0, pass_rhat=0, pass_rhat105=0, pass_rhat_and_rstar=0)
for f in sorted(glob.glob(os.path.join(R, 'silent_graph', '*.json'))):
    a = json.load(open(f))
    for row in a.get('rows', []):
        if row['dbeta'] <= 0:
            continue
        for gr in row['groups']:
            labs = [c['label'] for c in gr['chains'] if c['label'] != 'ambiguous']
            if gr['start'] == 'random' and labs and all(l == 'wrong' for l in labs):
                sil['wrong_random_groups_above'] += 1
                sil['pass_rhat'] += int(gr['rhat'] < 1.01)
                sil['pass_rhat105'] += int(gr['rhat'] < 1.05)
                sil['pass_rhat_and_rstar'] += int(gr['rhat'] < 1.01 and gr['rstar'] < 1.05)
out['graph'] = gg
out['graph_above'] = ga
out['graph_silent'] = sil
out['graph_instances'] = inst
json.dump(out, open(os.path.join(R, 'summary_silent.json'), 'w'), indent=1)

# exactly stationary controls on graphs: planted-start chains at the model-consistent beta (results/alarm)
ctl = new()
for f in sorted(glob.glob(os.path.join(R, 'alarm', '*.json'))):
    a = json.load(open(f))
    for gr in a['groups']:
        if gr['start'] == 'planted':
            add_group(ctl, gr, True, lambda c: True)
out['graph_exact_controls'] = ctl
out['graph_exact_controls_rhat_undefined'] = sum(1 for f in glob.glob(os.path.join(R, 'alarm', '*.json')) for gr in json.load(open(f))['groups'] if gr['start'] == 'planted' and gr['rhat'] != gr['rhat'])
json.dump(out, open(os.path.join(R, 'summary_silent.json'), 'w'), indent=1)

cell = lambda c: f"{c['fa']}/{c['nfa']}" if c['nfa'] else '--'
celd = lambda c: f"{c['det']}/{c['ndet']}" if c['ndet'] else '--'
lines = ['\\begin{tabular}{l@{\\hspace{5pt}}ccccc}', '\\toprule',
         ' & \\multicolumn{2}{c}{exact controls} & VS & \\multicolumn{2}{c}{graph (window)} \\\\',
         '\\cmidrule(lr){2-3}\\cmidrule(lr){4-4}\\cmidrule(lr){5-6}',
         'diagnostic & VS & graph & detect & occ.-corr. & detect \\\\', '\\midrule']
for key, lab, lev in DIAG:
    lines.append(f"{lab} & {cell(bv[key])} & {cell(ctl[key])} & {celd(bv[key])} & {cell(gg[key])} & {celd(gg[key])} \\\\")
    if key == 'dratio':
        lines.append('\\midrule')
lines += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(T, 'silent.tex'), 'w').write('\n'.join(lines) + '\n')
print(json.dumps({k: out[k] for k in ('bvs', 'graph_exact_controls', 'graph', 'graph_silent')}))
print(inst)
