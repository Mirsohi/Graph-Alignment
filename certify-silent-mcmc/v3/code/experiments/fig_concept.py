"""Figure 1 (overlap profiles; O.19 two-phase, O.16 multi-phase) and the BVS
Lambda(lambda) figure.  Profiles by stratified windows (2^21 per window)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.instances import load_instance
from gacert2.windows import overlap_profile
sys.path.insert(0, HERE)
from figstyle import STYLE, TEXTWIDTH
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
plt.rcParams.update(STYLE)
plt.rcParams.update({'lines.linewidth': 1.3})
cache = os.path.join(V2, 'results', 'profiles_fig1.json')
prof = json.load(open(cache)) if os.path.exists(cache) else {}
for case, betas in (('pscdb_O_19', (1.44, 1.463, 1.48)), ('pscdb_O_16', (1.216, 1.30))):
    A, B, pt, w, meta = load_instance(case)
    for b in betas:
        key = f'{case}_{b}'
        if key not in prof:
            prof[key] = overlap_profile(A, B, w, b, 2 ** 21, 3)['logp'].tolist()
            json.dump(prof, open(cache, 'w'))
fig, ax = plt.subplots(1, 2, figsize=(0.82 * TEXTWIDTH, 1.9))
for j, b in enumerate((1.44, 1.463, 1.48)):
    lp = np.array(prof[f'pscdb_O_19_{b}'])
    k = np.arange(len(lp))
    ax[0].plot(k[np.isfinite(lp)], lp[np.isfinite(lp)], color=C[j], label=fr'$\beta={b}$')
ax[0].axvline(31, color='0.4', ls=':', lw=0.8)
ax[0].text(15, -19, r'$D$', ha='center'); ax[0].text(60, -19, r'$C$', ha='center')
ax[0].set_ylim(-30, 2); ax[0].set_xlabel(r'overlap $k$ with witness'); ax[0].set_ylabel(r'$\log p_k$')
ax[0].set_title('O.19 ($n=103$): two phases')
ax[0].legend(loc='lower center', ncol=3, columnspacing=0.6, handlelength=0.9, handletextpad=0.3)
for j, b in enumerate((1.216, 1.30)):
    lp = np.array(prof[f'pscdb_O_16_{b}'])
    k = np.arange(len(lp))
    ax[1].plot(k[np.isfinite(lp)], lp[np.isfinite(lp)], color=C[j + 3], label=fr'$\beta={b}$')
ax[1].axvline(46, color='0.4', ls=':', lw=0.8)
ax[1].annotate('partially aligned phase', xy=(34, -3), xytext=(70, -12), fontsize=7, arrowprops=dict(arrowstyle='->', lw=0.6))
ax[1].set_ylim(-40, 2); ax[1].set_xlabel(r'overlap $k$ with witness'); ax[1].set_ylabel(r'$\log p_k$')
ax[1].set_title('O.16 ($n=154$): a third phase inside $D$')
ax[1].legend(loc='lower left')
fig.tight_layout()
fig.savefig(os.path.join(V2, 'figures', 'profiles.pdf')); fig.savefig(os.path.join(V2, 'figures', 'profiles.png'), dpi=150)
# BVS Lambda figure
fig, ax = plt.subplots(1, 2, figsize=(0.75 * TEXTWIDTH, 1.75))
for j, (nm, title) in enumerate((('exact20_strong', 'suppressor block, $p=20$'), ('decoy20', 'decoy block, $p=20$'))):
    d = json.load(open(os.path.join(V2, 'results', 'bvs', nm + '.json')))
    lam = np.array(d['ti']['lams'])
    a = ax[j]
    a.plot(lam, d['exact']['Lambda'], color='k', lw=2.2, alpha=0.35, label='exact')
    a.plot(lam, d['ti']['Lambda_fp'], color=C[1], label='Friel–Pettitt TI')
    a.plot(lam, d['ti']['Lambda_2ph'], color=C[0], ls='--', label='two-phase TI')
    a.plot(lam, d['ti']['witness_floor'], color=C[2], ls=':', label='witness floor')
    a.set_xlabel(r'$\lambda$'); a.set_title(title)
    if j == 0:
        a.set_ylabel(r'$\Lambda(\lambda)=\log\sum\rho L^\lambda$')
        a.legend(loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(V2, 'figures', 'bvs_lambda.pdf')); fig.savefig(os.path.join(V2, 'figures', 'bvs_lambda.png'), dpi=150)
print('ok')
