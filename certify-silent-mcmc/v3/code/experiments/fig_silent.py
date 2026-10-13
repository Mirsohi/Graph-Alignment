"""Figure: silent regimes.  Each dot is one group of four chains: split-R-hat (x)
vs the mean two-phase TV lower bound of its chains (y), coloured by the GT label
(exact posterior for variable selection, window GT for graph alignment).
Writes figures/silent.pdf/png."""
import json, os, glob, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
R = os.path.join(V2, 'results')
COL = {'correct': '#2a78d6', 'wrong': '#eb6834'}
MK = {'correct': 'o', 'wrong': '^'}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import STYLE, TEXTWIDTH
plt.rcParams.update(STYLE)


def glabel(gr, exact_start=False):
    if exact_start:
        return 'correct'
    labs = [c['label'] for c in gr['chains'] if c['label'] != 'ambiguous']
    if labs and len(labs) == len(gr['chains']) and all(l == 'correct' for l in labs):
        return 'correct'
    if labs and all(l == 'wrong' for l in labs):
        return 'wrong'
    return None


pts = {'bvs': [], 'graph': []}
for f in sorted(glob.glob(os.path.join(R, 'silent_bvs', '*.json'))):
    a = json.load(open(f))
    for gr in a['groups']:
        lab = glabel(gr, gr['start'] == 'exact')
        if lab is None or (gr['start'] != 'exact' and lab == 'correct'):
            continue    # false-alarm population in BVS = exact starts only
        tv = np.mean([abs(c['nuC'] - a['wC_2p']) for c in gr['chains']])
        pts['bvs'].append((gr['rhat'], tv, lab))
for f in sorted(glob.glob(os.path.join(R, 'silent_graph', '*.json'))):
    a = json.load(open(f))
    for row in a.get('rows', []):
        for gr in row['groups']:
            lab = glabel(gr)
            if lab is None:
                continue
            tv = np.mean([c['tv_est'] for c in gr['chains']])
            pts['graph'].append((gr['rhat'], tv, lab))
fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.1), sharey=True)
rng = np.random.default_rng(0)
for a_, (key, title) in zip(ax, (('bvs', 'variable selection (exact labels)'),
                                 ('graph', 'graph alignment (window-reference labels)'))):
    for lab in ('correct', 'wrong'):
        P = np.array([(x, y) for x, y, l in pts[key] if l == lab])
        if not len(P):
            continue
        x = np.clip(P[:, 0], 0.995, 3.0) - 1.0
        y = P[:, 1] + rng.uniform(-0.012, 0.012, len(P))
        n = len(P)
        low = x < 1e-3
        x = np.where(low, 10 ** rng.uniform(np.log10(4e-4), np.log10(1e-3), len(x)), x)
        a_.scatter(x, y, s=14, marker=MK[lab], color=COL[lab], edgecolor='white', linewidth=0.5,
                   alpha=0.9, label=f'{lab} groups', zorder=3)
    a_.set_xscale('log')
    a_.set_xlim(3e-4, 2.5)
    a_.axvline(0.01, color='0.45', ls=':', lw=0.8)
    a_.axhline(0.5, color='0.45', ls=':', lw=0.8)
    a_.text(0.0105, 1.04, r'$\widehat R=1.01$', fontsize=7, color='0.35')
    a_.set_xticks([7e-4, 1e-2, 1e-1, 1])
    a_.set_xticklabels(['$\\leq$1.001', '1.01', '1.1', '2'])
    a_.set_xlabel(r'split-$\widehat R$ of the group')
    a_.set_title(title)
    a_.set_ylim(-0.05, 1.12)
ax[0].set_ylabel('two-phase occupancy error\n(mean over chains)')
ax[1].legend(loc='center right', bbox_to_anchor=(1.0, 0.3))
fig.tight_layout()
fig.savefig(os.path.join(V2, 'figures', 'silent.pdf'))
fig.savefig(os.path.join(V2, 'figures', 'silent.png'), dpi=150)
print({k: len(v) for k, v in pts.items()})
