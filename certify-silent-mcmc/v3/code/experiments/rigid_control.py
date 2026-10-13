"""Compact-patch control for the partially aligned phases (critic round 2):
for each centre residue, the patch of its k nearest residues in structure A;
self-RMSD after superposition on the patch.  Compared with the aligned set.
Also the AdK per-residue agreement plot with CORE/NMP/LID shading.
Writes results/middle_phase/compact_control.json and figures/adk_partial.pdf."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import STYLE, TEXTWIDTH
plt.rcParams.update(STYLE)
V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(V2)


def self_rmsd(X, Y, sel):
    Xc = X[sel] - X[sel].mean(0); Yc = Y[sel] - Y[sel].mean(0)
    U, S, Vt = np.linalg.svd(Xc.T @ Yc); d = np.sign(np.linalg.det(U @ Vt)); Rm = U @ np.diag([1, 1, d]) @ Vt
    return float(np.sqrt(np.mean(np.sum((Xc @ Rm - Yc) ** 2, 1))))


out = {}
for c, b in [('adk_open_closed', 1.961), ('pscdb_O_16', 1.216), ('pscdb_IL_74', 1.775)]:
    m = json.load(open(os.path.join(V2, 'results', 'middle_phase', f'{c}_{b}.json')))
    z = np.load(os.path.join(ROOT, 'data', 'cases', c + '.npz'))
    X, Y = z['x'], z['y']
    n = len(X)
    F = np.array(m['agree_freq'])
    al = F > 0.5
    k = int(al.sum())
    r_al = self_rmsd(X, Y, al)
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    patch = []
    for i in range(n):
        sel = np.argsort(D[i])[:k]
        patch.append(self_rmsd(X, Y, sel))
    patch = np.array(patch)
    # spatial compactness of the aligned set: radius of gyration vs patches
    rg = lambda sel: float(np.sqrt(np.mean(np.sum((X[sel] - X[sel].mean(0)) ** 2, 1))))
    rg_al = rg(al)
    rg_patch = float(np.median([rg(np.argsort(D[i])[:k]) for i in range(n)]))
    rec = dict(k=k, aligned_rmsd=r_al, patch_median=float(np.median(patch)), patch_min=float(patch.min()),
               patch_frac_le=float(np.mean(patch <= r_al + 1e-12)), rg_aligned=rg_al, rg_patch_median=rg_patch)
    if c == 'adk_open_closed':
        core = np.zeros(n, bool)
        idx = np.arange(1, n + 1)
        core[((idx >= 1) & (idx <= 29)) | ((idx >= 60) & (idx <= 121)) | ((idx >= 160) & (idx <= 214))] = True
        rec['frac_aligned_in_CORE'] = float(core[al].mean())
        rec['core_self_rmsd'] = self_rmsd(X, Y, core)
        fig, ax = plt.subplots(figsize=(0.6 * TEXTWIDTH, 1.8))
        for lo, hi, lab, col in ((30, 59, 'NMP', '#eda100'), (122, 159, 'LID', '#eb6834')):
            ax.axvspan(lo - 0.5, hi + 0.5, color=col, alpha=0.18, lw=0)
            ax.text((lo + hi) / 2, 1.05, lab, ha='center', fontsize=7)
        ax.text(95, 1.05, 'CORE', ha='center', fontsize=7)
        ax.bar(idx, F, width=1.0, color='#2a78d6')
        ax.set_xlim(0.5, n + 0.5); ax.set_ylim(0, 1.15)
        ax.set_xlabel('residue'); ax.set_ylabel('agreement')
        fig.tight_layout()
        fig.savefig(os.path.join(V2, 'figures', 'adk_partial.pdf'))
        fig.savefig(os.path.join(V2, 'figures', 'adk_partial.png'), dpi=150)
    out[c] = rec
    print(c, rec)
json.dump(out, open(os.path.join(V2, 'results', 'middle_phase', 'compact_control.json'), 'w'), indent=1)
