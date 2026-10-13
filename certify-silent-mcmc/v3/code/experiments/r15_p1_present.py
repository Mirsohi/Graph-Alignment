"""R15 M2 presentation (not part of the frozen analysis): reads results/r15/p1_heldout/summary_p1.json and writes
paper/tables/r15_p1.tex, paper/tables/r15_p1_family.tex and figures/r15_p1_accuracy_coverage.pdf (draft).
All numbers are EMPIRICAL (Monte Carlo pipeline vs window reference); nothing here is a guarantee.
Usage: python r15_p1_present.py [p1_dir]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from r15_common import V3
from r15_p1_common import P1
from figstyle import STYLE, TEXTWIDTH
plt.rcParams.update(STYLE)

LAB = {'O0': 'O0 no screen', 'O1': 'O1 first attempt', 'O2': 'O2 released (primary)', 'O2_t0.25': r'O2, $t{=}0.25$',
       'O2_t0.5': r'O2, $t{=}0.5$', 'O2_t2.0': r'O2, $t{=}2$', 'wl_ext': 'Wang--Landau (matched)', 'pt_broad_w': 'witness PT (matched)'}


def f(x, d=2):
    return '--' if x is None else f'{x:.{d}f}'


def ci(c):
    return '' if not c else f' [{c[0]:.2f}, {c[1]:.2f}]'


def table(S):
    L = [r'% R15 M2 draft (r15_p1_present.py from results/r15/p1_heldout/summary_p1.json). EMPIRICAL.',
         f"% {S['n_instances_complete']} complete held-out instances ({len(S['runs'])} pipeline runs); unit = run (pipeline), instance (comparators).",
         r'\begin{tabular}{lcccccc}', r'\toprule',
         r'operating point & coverage & false-pass & false-pass & $e$ median & $e$ 90\% & $\log_2$ proposals \\',
         r' & & $\varepsilon{=}0.05$ & $\varepsilon{=}0.10$ & (returned) & (returned) & per useful output \\', r'\midrule']
    for op in ('O0', 'O1', 'O2', 'O2_t0.25', 'O2_t0.5', 'O2_t2.0'):
        o = S['ops'][op]
        cpu = o['eps0.05']['cost_per_useful']
        L.append(f"{LAB[op]} & {f(o['coverage'])}{ci(o.get('ci95', {}).get('coverage'))} & "
                 f"{o['eps0.05']['false_pass']}/{o['n_returned']}{ci(o.get('ci95', {}).get('fpr_eps005'))} & "
                 f"{o['eps0.1']['false_pass']}/{o['n_returned']} & {f(o['e_returned']['median'], 3)} & {f(o['e_returned']['q90'], 3)} & "
                 f"{f(np.log2(cpu) if cpu else None, 1)} \\\\")
    L.append(r'\midrule')
    for m in ('wl_ext', 'pt_broad_w'):
        o = S['comparators'].get(m)
        if not o:
            continue
        cpu = o['eps0.05']['cost_per_useful']
        L.append(f"{LAB[m]} & 1 & {o['eps0.05']['false_pass']}/{o['n_returned']} & {o['eps0.1']['false_pass']}/{o['n_returned']} & "
                 f"{f(o['e_returned']['median'], 3)} & {f(o['e_returned']['q90'], 3)} & {f(np.log2(cpu) if cpu else None, 1)} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(L) + '\n'


def family_table(S):
    L = [r'% R15 M2 per-family draft (EMPIRICAL). cov = coverage at O2; fp = false passes at eps=0.05 / returned.',
         r'\begin{tabular}{lcccccc}', r'\toprule',
         r'family & runs & O0 fp & O2 cov & O2 fp & WL fp & PT fp \\', r'\midrule']
    for fam, d in sorted(S['per_family'].items()):
        o0, o2 = d['O0'], d['O2']
        wl = S['comparators'].get('wl_ext', {}).get('per_family', {}).get(fam)
        pt = S['comparators'].get('pt_broad_w', {}).get('per_family', {}).get(fam)
        L.append(f"{fam} & {o0['n_runs']} & {o0['eps0.05']['false_pass']}/{o0['n_returned']} & {f(o2['coverage'])} & "
                 f"{o2['eps0.05']['false_pass']}/{o2['n_returned']} & "
                 f"{(str(wl['eps0.05']['false_pass']) + '/' + str(wl['n_returned'])) if wl else '--'} & "
                 f"{(str(pt['eps0.05']['false_pass']) + '/' + str(pt['n_returned'])) if pt else '--'} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(L) + '\n'


def figure(S, path):
    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.3))
    a = ax[0]
    pts = ['O0', 'O1', 'O2_t0.25', 'O2_t0.5', 'O2', 'O2_t2.0']
    for key, mk, col, lab in (('fpr', 'o', '#eb6834', r'false-pass rate ($\varepsilon{=}0.05$)'),
                              ('q90', 's', '#2a78d6', r'90\% error of returned runs')):
        xs, ys, ls = [], [], []
        for op in pts:
            o = S['ops'][op]
            y = o['eps0.05']['fpr'] if key == 'fpr' else o['e_returned']['q90']
            if y is None:
                continue
            xs.append(o['coverage']); ys.append(y); ls.append(op)
        a.plot(xs, ys, mk, color=col, ms=4, label=lab.replace(r'\%', '%'))
        for x, y, l in zip(xs, ys, ls):
            dy = -8 if l == 'O1' else 2
            a.annotate(l.replace('O2_t', 't='), (x, y), fontsize=6, xytext=(3, dy), textcoords='offset points', color='0.3')
    for m, mk in (('wl_ext', 'v'), ('pt_broad_w', '^')):
        o = S['comparators'].get(m)
        if o:
            a.plot([1.0], [o['eps0.05']['fpr']], mk, mfc='none', color='#eb6834', ms=5)
            a.plot([1.0], [o['e_returned']['q90']], mk, mfc='none', color='#2a78d6', ms=5)
            a.annotate({'wl_ext': 'WL', 'pt_broad_w': 'PT'}[m], (1.0, o['eps0.05']['fpr']), fontsize=6,
                       xytext=({'wl_ext': -14, 'pt_broad_w': 4}[m], -9 if m == 'wl_ext' else 2), textcoords='offset points', color='0.3')
    a.axhline(0.05, color='0.6', ls=':', lw=0.7)
    a.set_xlabel('coverage (fraction of runs returned)')
    a.set_ylabel('rate / error')
    a.set_xlim(-0.03, 1.08)
    a.set_title('(a) accuracy vs coverage (empirical)')
    a.legend(loc='upper left', fontsize=6)
    b = ax[1]
    runs = S['runs']
    for ret, col, lab in ((True, '#2a78d6', 'pipeline, returned (O2)'), (False, '0.6', 'pipeline, abstained')):
        R = [r for r in runs if r['O2'] == ret]
        b.scatter([np.log2(r['cost']) for r in R], [r['e'] for r in R], s=10, color=col, label=lab, zorder=3,
                  edgecolor='white', linewidth=0.4)
    for m, mk, col in (('wl_ext', 'v', '#7a5195'), ('pt_broad_w', '^', '#ef8f00')):
        o = S['comparators'].get(m)
        if o:
            b.scatter([np.log2(r['cost']) for r in o['instances']], [r['e'] for r in o['instances']], s=10, marker=mk,
                      color=col, label={'wl_ext': 'Wang--Landau', 'pt_broad_w': 'witness PT'}[m], zorder=2)
    b.axhline(0.05, color='0.6', ls=':', lw=0.7)
    b.set_yscale('log')
    b.set_ylim(5e-4, 3)
    b.set_xlabel(r'total proposals per run ($\log_2$)')
    b.set_ylabel(r'$e=\max_{\beta\in G}|\hat a-a_{\rm ref}|$')
    b.set_title('(b) error vs total compute', pad=22)
    b.ticklabel_format(axis='x', useOffset=False, style='plain')
    secs = np.array([r['sec'] for r in runs]); cost = np.array([r['cost'] for r in runs])
    if len(runs):
        rate = float(np.median(cost / secs))
        top = b.secondary_xaxis('top', functions=(lambda x: 2 ** x / rate, lambda s: np.log2(np.maximum(s, 1e-9) * rate)))
        top.set_xlabel('wall-clock s (median throughput)', fontsize=6, labelpad=1)
        top.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))
    b.legend(loc='lower right', fontsize=5.5, framealpha=0.85, frameon=True, edgecolor='none')
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'), dpi=150)


if __name__ == '__main__':
    p1 = sys.argv[1] if len(sys.argv) > 1 else P1
    S = json.load(open(os.path.join(p1, 'summary_p1.json')))
    test = p1 == P1
    tdir = os.path.join(V3, 'paper', 'tables') if test else p1
    fdir = os.path.join(V3, 'figures') if test else p1
    open(os.path.join(tdir, 'r15_p1.tex'), 'w').write(table(S))
    open(os.path.join(tdir, 'r15_p1_family.tex'), 'w').write(family_table(S))
    figure(S, os.path.join(fdir, 'r15_p1_accuracy_coverage.pdf'))
    print('wrote tables and figure to', tdir, fdir)
