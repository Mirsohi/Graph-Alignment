"""Random-subset control for middle-phase rigidity (self-RMSD after superposition on
k random residues vs on the k aligned residues).  Writes results/middle_phase/control.json."""
import json, os, numpy as np
V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def dev(X, Y, sel):
    Xc = X[sel] - X[sel].mean(0); Yc = Y[sel] - Y[sel].mean(0)
    U, S, Vt = np.linalg.svd(Xc.T @ Yc); d = np.sign(np.linalg.det(U @ Vt)); Rm = U @ np.diag([1, 1, d]) @ Vt
    return np.linalg.norm((X - X[sel].mean(0)) @ Rm + Y[sel].mean(0) - Y, axis=1)
rng = np.random.default_rng(0); out = {}
for c, b in [('adk_open_closed', 1.961), ('pscdb_O_16', 1.216), ('pscdb_IL_74', 1.775)]:
    m = json.load(open(os.path.join(V2, 'results', 'middle_phase', f'{c}_{b}.json')))
    z = np.load(os.path.join(os.path.dirname(V2), 'data', 'cases', c + '.npz')); X, Y = z['x'], z['y']; n = len(X)
    k = int((np.array(m['agree_freq']) > 0.5).sum())
    r = []
    for _ in range(1000):
        sel = rng.choice(n, k, replace=False); r.append(float(np.sqrt(np.mean(dev(X, Y, sel)[sel] ** 2))))
    out[c] = dict(k=k, aligned_self_rmsd=m['rmsd_aligned_after_fit_on_aligned'], rest_rmsd=m['rmsd_rest_after_fit_on_aligned'],
                  random_median=float(np.median(r)), random_p05=float(np.percentile(r, 5)),
                  frac_random_below=float(np.mean(np.array(r) <= m['rmsd_aligned_after_fit_on_aligned'])),
                  segments=m['segments'], beta=b)
    print(c, out[c])
json.dump(out, open(os.path.join(V2, 'results', 'middle_phase', 'control.json'), 'w'), indent=1)
