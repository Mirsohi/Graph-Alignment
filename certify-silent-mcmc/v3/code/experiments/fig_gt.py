"""Figure: condensed-phase weight and marginal error vs ground truth on the
instances with ground truth.  Reads results/gt_comparison.json and twophase/*_gt.json.
Usage: python fig_gt.py [case1 case2 ...] -> v2/figures/gt_phase.pdf
(default: the five instances of Figure 3).  Round 10: the shared legend sits at the top
of the canvas so that it cannot collide with the LaTeX caption.
Round 15: --top writes only the phase-weight row (main-text Figure 3) to figures/gt_phase_top.pdf;
without it the two-row figure (phase weight and row-TV, appendix) is written to figures/gt_phase.pdf."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
R = os.path.join(V2, 'results')
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import STYLE, TEXTWIDTH
plt.rcParams.update(STYLE)
from matplotlib.ticker import MaxNLocator
TOP = '--top' in sys.argv
cases = [a for a in sys.argv[1:] if not a.startswith('--')] or ['pscdb_N_74', 'pscdb_CL_64', 'pscdb_O_19', 'syn_n40_d6_q0.1_s1', 'pscdb_N_141']
G = json.load(open(os.path.join(R, 'gt_comparison.json')))
fig, ax = plt.subplots(1 if TOP else 2, len(cases), figsize=(0.94 * TEXTWIDTH, 1.45 if TOP else 2.65), squeeze=False)
meth = [('mh_random', 'MH random start', 'v'), ('mh_witness', 'MH witness start', '^'),
        ('annealed', 'annealed MH', 's'), ('pt', 'parallel tempering', 'D')]
for j, case in enumerate(cases):
    d = json.load(open(os.path.join(R, 'twophase', f'{case}_gt.json')))
    g = G[case]
    bs = np.array([r['beta'] for r in g['rows']])
    b = np.linspace(bs.min() - 0.01, bs.max() + 0.01, 400)
    W = []
    for r in d['reps']:
        dF = np.interp(b, d['gridC'], r['FC']) - np.interp(b, d['gridD'], r['FD'])
        W.append(1 / (1 + np.exp(-dF)))
    W = np.array(W)
    a = ax[0, j]
    a.fill_between(b, W.min(0), W.max(0), color=C[0], alpha=0.25, lw=0)
    a.plot(b, W.mean(0), color=C[0], label='two-phase TI (ours)')
    for k, (m, lab, mk) in enumerate(meth):
        a.plot(bs, [r[m]['w'] for r in g['rows']], mk, ms=3.5, mfc='none', color=C[k + 1], label=lab)
    a.plot(bs, [r['w_gt'] for r in g['rows']], 'o', ms=3, color='k', label='reference')
    a.set_title(('synth.' if case.startswith('syn') else case.replace('pscdb_', '').replace('_', '.')) + f" ($n$={g['n']})")
    if j == 0:
        a.set_ylabel(r'$\omega_{\rm C}(\beta)$')
    a.set_xlabel(r'$\beta$') if TOP else None
    if TOP:
        a.xaxis.set_major_locator(MaxNLocator(2)); a.yaxis.set_major_locator(MaxNLocator(3))
        continue
    a2 = ax[1, j]
    a2.plot(bs, [r['twophase']['tv'] for r in g['rows']], '-o', ms=3, color=C[0])
    for k, (m, lab, mk) in enumerate(meth):
        a2.plot(bs, [r[m]['tv'] for r in g['rows']], mk, ms=3.5, mfc='none', color=C[k + 1])
    a2.plot(bs, [r['gt_floor_tv'] for r in g['rows']], ':', color='0.5', label='reference replicate floor')
    a2.set_xlabel(r'$\beta$')
    for a_ in (a, a2):
        a_.xaxis.set_major_locator(MaxNLocator(2))
        a_.yaxis.set_major_locator(MaxNLocator(3))
    if j == 0:
        a2.set_ylabel('row-TV')
h1, l1 = ax[0, 0].get_legend_handles_labels()
os.makedirs(os.path.join(V2, 'figures'), exist_ok=True)
if TOP:
    fig.legend(h1, l1, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=6, columnspacing=0.6, handlelength=1.1, handletextpad=0.3)
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.25, top=0.72, wspace=0.45)
    fig.savefig(os.path.join(V2, 'figures', 'gt_phase_top.pdf'))
    fig.savefig(os.path.join(V2, 'figures', 'gt_phase_top.png'), dpi=150)
else:
    h2, l2 = ax[1, 0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, columnspacing=1.0, handlelength=1.5)
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.15, top=0.78, wspace=0.45, hspace=0.5)
    fig.savefig(os.path.join(V2, 'figures', 'gt_phase.pdf'))
    fig.savefig(os.path.join(V2, 'figures', 'gt_phase.png'), dpi=150)
