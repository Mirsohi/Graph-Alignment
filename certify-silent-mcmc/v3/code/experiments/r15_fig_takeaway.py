"""R15 S4 takeaway figure (draft; same footprint as the current Fig. 2, 0.75 textwidth x 1.75 in).
Left: p=20 suppressor lambda-path (exact Lambda, Friel-Pettitt TI, two-phase TI, witness floor) from results/bvs.
Right: O.19 along beta, plotted relative to the point floor, Lambda(beta) - beta S(w):
  two-phase Lambda (TI-BASED, released rep 0), FP-TI paths (M1, 2^17 and 2^21, seed 101), point floor (= 0),
  coset-ELBO floor (RIGOROUS) if --coset, shaded certified error (floor above FP-TI), nucleation marked.
Writes figures/r15_takeaway.pdf/png.  Usage: python r15_fig_takeaway.py [--coset]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from r15_common import V3, RES
from figstyle import STYLE, TEXTWIDTH
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#7a5195']
plt.rcParams.update(STYLE)
plt.rcParams.update({'lines.linewidth': 1.2})
CASE = 'pscdb_O_19'

fig, ax = plt.subplots(1, 2, figsize=(0.94 * TEXTWIDTH, 1.6))  # R15 paper version: full figure* width, fonts >= 6.5 pt
d = json.load(open(os.path.join(V3, 'results', 'bvs', 'exact20_strong.json')))
lam = np.array(d['ti']['lams'])
a = ax[0]
a.plot(lam, d['exact']['Lambda'], color='k', lw=2.2, alpha=0.35, label='exact')
a.plot(lam, d['ti']['Lambda_fp'], color=C[1], label='standard TI')
a.plot(lam, d['ti']['Lambda_2ph'], color=C[0], ls='--', label='two-phase TI')
a.plot(lam, d['ti']['witness_floor'], color=C[2], ls=':', label='witness floor')
fl, fp = np.array(d['ti']['witness_floor']), np.array(d['ti']['Lambda_fp'])
a.fill_between(lam, fp, fl, where=fl > fp, color=C[1], alpha=0.15, lw=0)
a.set_xlabel(r'$\lambda$'); a.set_title('variable selection (suppressor, $p=20$)')
a.set_ylabel(r'$\Lambda$ (nats)')
a.legend(loc='upper left', fontsize=6.5, handlelength=1.6, labelspacing=0.2)
a.set_ylim(top=60)

b = ax[1]
tp = json.load(open(os.path.join(V3, 'results', 'twophase', CASE + '_all.json')))
Sw = tp['meta']['Sw']
r0 = tp['reps'][0]
bD, FD = np.array(tp['gridD']), np.array(r0['FD'])
bC, FC = np.array(tp['gridC']), np.array(r0['FC'])
if bC[0] > bC[-1]:
    bC, FC = bC[::-1], FC[::-1]
g = np.round(np.arange(1.2, 3.0001, 0.01), 4)
fD = np.where(g <= bD[-1], np.interp(g, bD, FD), FD[-1] + (g - bD[-1]) * np.array(r0['meanS_D'])[-1])
L2 = np.logaddexp(np.interp(g, bC, FC), fD)
b.plot(g, L2 - g * Sw, color=C[0], ls='--', label='two-phase (TI-based)')
b.axhline(0, color=C[2], ls=':', lw=1.2, label=r'witness floor $\beta S(w)$')
ymin = -10
for logT, col, ls in ((17, '0.55', '-'), (21, C[1], '-')):
    p = os.path.join(RES, 'fpti_cert', f'{CASE}_T{logT}_s101.json')
    if not os.path.exists(p):
        continue
    r = json.load(open(p))
    x, L = np.array(r['path']['beta']), np.array(r['path']['Lambda_FP'])
    m = x >= 1.2
    y = L[m] - x[m] * r['S_star']
    b.plot(x[m], y, color=col, ls=ls, label=fr'standard TI $2^{{{logT}}}$/pt')
    ymin = min(ymin, float(y.min()))
    if logT == 21:
        b.fill_between(x[m], y, 0, where=y < 0, color=C[1], alpha=0.35, lw=0)
        nb = r['nucleation_beta']
        if nb is not None:
            yn = float(np.interp(nb, x, L - x * r['S_star']))
            b.plot([nb], [yn], 'o', color=C[1], ms=3)
            b.annotate('nucleation', (nb, yn), xytext=(nb + 0.2, yn - 19), fontsize=6.5,
                       arrowprops=dict(arrowstyle='->', lw=0.5))
if '--coset' in sys.argv:
    mf = json.load(open(os.path.join(RES, 'mass_floor', CASE + '.json')))
    xb = np.array([q['beta'] for q in mf['rows']]); yc = np.array([q['coset'] - q['point'] for q in mf['rows']])
    o = np.argsort(xb)
    b.plot(xb[o], yc[o], color=C[5], ls='-.', label='coset-ELBO floor')
b.set_xlabel(r'$\beta$'); b.set_ylabel(r'$\Lambda(\beta)-\beta S(w)$ (nats)')
b.set_title('graph alignment (O.19, $n=103$)')
b.set_ylim(1.15 * ymin, 40)
h, l = b.get_legend_handles_labels()
b.legend(h, l, loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=6.5, handlelength=1.6, borderpad=0.1, labelspacing=0.25)
fig.tight_layout()
fig.subplots_adjust(right=0.80, wspace=0.45)
fig.savefig(os.path.join(V3, 'figures', 'r15_takeaway.pdf'))
fig.savefig(os.path.join(V3, 'figures', 'r15_takeaway.png'), dpi=150)
print('ok')
