"""I4: barrier-based bounds from the overlap free-energy profile p_k = mu_beta(k).

For random-transposition MH and any level k0 (D = {k < k0}, C = {k >= k0}):
  Q(D,C) <= r_up(k0) * mu(k in {k0-2, k0-1}),  r_up(k0) = 2 (n - k0 + 2) / (n (n-1)),
  Q(C,D) <= r_dn(k0) * mu(k in {k0, k0+1}),    r_dn(k0) = 2 (k0 + 1) / n,
and Q(D,C) = Q(C,D) (reversibility), so
  (a) E[# full phase switches (k<=klo -> k>=khi or back) in T stationary steps]
        <= 2 T min_{klo < k0 <= khi} min(r_up mu(up-shell), r_dn mu(down-shell));
  (b) P_{mu|D}(tau_C <= t) <= t * min(...)/mu(D)   (escape from the diffuse phase),
      P_{mu|C}(tau_D <= t) <= t * min(...)/mu(C).
Profiles come from the umbrella ground-truth runs (exact reweighting).
Compares (a) with switch counts observed in long unrestricted MH runs."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
R = os.path.join(V2, 'results')


def bounds(pk, n, klo, khi):
    best_flux, best_k = np.inf, None
    for k0 in range(max(klo + 1, 2), khi + 1):
        up = 2 * (n - k0 + 2) / (n * (n - 1)) * (pk[k0 - 2] + pk[k0 - 1])
        dn = 2 * (k0 + 1) / n * (pk[k0] + (pk[k0 + 1] if k0 + 1 < len(pk) else 0))
        f = min(up, dn)
        if f < best_flux:
            best_flux, best_k = f, k0
    muD = pk[:best_k].sum()
    return dict(flux_bound=float(best_flux), k_bottleneck=int(best_k),
                max_switch_rate=float(2 * best_flux),
                escape_D_steps_half=float(0.5 * muD / best_flux) if best_flux > 0 else np.inf,
                escape_C_steps_half=float(0.5 * (1 - muD) / best_flux) if best_flux > 0 else np.inf,
                barrier_nats=float(-np.log(max(pk[best_k - 2] + pk[best_k - 1], 1e-300)) +
                                   np.log(max(min(muD, 1 - muD), 1e-300))))


def main(cases):
    out = {}
    for case in cases:
        u = json.load(open(os.path.join(R, 'gt', f'gt_umb_{case}.json')))
        n = u['n']
        klo, khi = int(0.15 * n), int(0.5 * n)
        mh = None
        p = os.path.join(R, 'gt', f'gt_mh_{case}.json')
        if os.path.exists(p):
            mh = json.load(open(p))
        sw_p = os.path.join(R, 'gt', f'switch_{case}.json')
        sw = json.load(open(sw_p)) if os.path.exists(sw_p) else None
        rows = []
        for b, reps in u['betas'].items():
            pk = np.mean([np.array(r['pk']) for r in reps], 0)
            bd = bounds(pk, n, klo, khi)
            row = dict(beta=float(b), **bd)
            if mh and b in mh['betas']:
                obs = [r['switches'] / mh['T'] for r in mh['betas'][b]]
                row['observed_switch_rate_mh'] = obs
                row['T_mh'] = mh['T']
            elif sw and b in sw['betas']:
                row['observed_switch_rate_mh'] = [x / sw['T'] for x in sw['betas'][b]]
                row['T_mh'] = sw['T']
            rows.append(row)
            o = row.get('observed_switch_rate_mh')
            print(case, b, f"bound rate {bd['max_switch_rate']:.3e}  observed {o}  "
                  f"escape_D>= {bd['escape_D_steps_half']:.2e} escape_C>= {bd['escape_C_steps_half']:.2e} "
                  f"k*={bd['k_bottleneck']} barrier {bd['barrier_nats']:.1f}", flush=True)
        out[case] = dict(n=n, klo=klo, khi=khi, rows=rows)
    json.dump(out, open(os.path.join(R, 'barrier_bounds.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
