"""Figure: per-protein two-phase beta_c vs v1 heuristic beta-hat_c, with the
estimator's metastability window [C-spinodal, D-spinodal] and the v1 hysteresis
(oracle-start condensed from / random-start condensed from, on v1's coarse grid).
Usage: python fig_all.py [tag] -> v2/figures/all_betac.pdf"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
tag = sys.argv[1] if len(sys.argv) > 1 else 'all'
S = json.load(open(os.path.join(V2, 'results', f'summary_{tag}.json')))
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.6, 'legend.frameon': False, 'pdf.fonttype': 42})
rows = sorted(S['rows'], key=lambda r: r['beta_c'])
y = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(3.4, 4.8))
for i, r in enumerate(rows):
    ax.plot([r['spinodal_C'], r['spinodal_D']], [i, i], color=C[0], alpha=0.35, lw=4, solid_capstyle='butt',
            label='metastability window (ours)' if i == 0 else None)
ax.plot([r['beta_c'] for r in rows], y, 'o', color=C[0], ms=4, label=r'$\beta_c$ (two-phase, ours)')
ax.plot([r['beta_c_v1'] for r in rows], y, 'x', color=C[1], ms=4, label=r'$\hat\beta_c$ (v1 heuristic)')
oc = [r['v1_oracle_condensed_from'] if r['v1_oracle_condensed_from'] else np.nan for r in rows]
rc = [r['v1_random_condensed_from'] if r['v1_random_condensed_from'] else np.nan for r in rows]
ax.plot(oc, y - 0.25, '<', color=C[2], ms=3.5, mfc='none', label='v1 oracle start: condensed')
ax.plot(rc, y + 0.25, '>', color=C[3], ms=3.5, mfc='none', label='v1 random start: condensed')
ax.set_yticks(y)
ax.set_yticklabels([r['case'].replace('pscdb_', '').replace('_', '.').replace('adk.', 'AdK ') for r in rows], fontsize=6)
ax.set_xlabel(r'$\beta$')
ax.legend(loc='lower center', bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=5.5)
fig.tight_layout()
fig.savefig(os.path.join(V2, 'figures', 'all_betac.pdf'))
fig.savefig(os.path.join(V2, 'figures', 'all_betac.png'), dpi=150)
