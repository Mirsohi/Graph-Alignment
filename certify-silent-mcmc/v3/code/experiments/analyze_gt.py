"""Compare the two-phase estimator and matched-budget baselines with ground
truth (long MH or umbrella) on instances where ground truth is available.
Writes v2/results/gt_comparison.json and prints tables.
Usage: python analyze_gt.py case1:source1 case2:source2 ...  (source = mh|umb)"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
R = os.path.join(V2, 'results')


def rowtv(P, Q):
    return float(np.abs(P - Q).sum() / (2 * P.shape[0]))


def logit(x, eps=1e-4):
    x = min(max(x, eps), 1 - eps)
    return float(np.log(x / (1 - x)))


def load_gt(case, src):
    if src == 'mh':
        d = json.load(open(os.path.join(R, 'gt', f'gt_mh_{case}.json')))
        M = np.load(os.path.join(R, 'gt', f'gt_mh_{case}_marg.npz'))
        out = {}
        for b, rows in d['betas'].items():
            fr = [r['fracC'] for r in rows]
            Ms = [M[f'{float(b)}_{r["start"]}'] for r in rows]
            out[float(b)] = dict(w=float(np.mean(fr)), w_spread=float(abs(fr[0] - fr[1])),
                                 P=np.mean(Ms, 0), floor=rowtv(Ms[0], Ms[1]),
                                 switches=[r['switches'] for r in rows])
        return out
    pre = 'gt_win_' if src.startswith('win') else 'gt_umb_'
    suf = '_4s' if src == 'win4' else ''
    d = json.load(open(os.path.join(R, 'gt', f'{pre}{case}{suf}.json')))
    M = np.load(os.path.join(R, 'gt', f'{pre}{case}{suf}_marg.npz'))
    out = {}
    for b, rows in d['betas'].items():
        fr = [r['fracC'] for r in rows]
        Ms = [M[f'{float(b)}_s{s}'] for s in range(len(rows))]
        fl = [rowtv(Ms[i], Ms[j]) for i in range(len(Ms)) for j in range(i + 1, len(Ms))]
        out[float(b)] = dict(w=float(np.mean(fr)), w_spread=float(max(fr) - min(fr)),
                             w_se=float(np.std(fr, ddof=1) / np.sqrt(len(fr))),
                             P=np.mean(Ms, 0), floor=float(np.mean(fl)),
                             switches=[r['traversals'] for r in rows])
    return out


def main(specs):
    allres = {}
    for spec in specs:
        case, src = spec.split(':')
        gt = load_gt(case, src)
        est = json.load(open(os.path.join(R, 'twophase', f'{case}_gt.json')))
        EM = np.load(os.path.join(R, 'twophase', f'{case}_gt_marg.npz'))
        base = json.load(open(os.path.join(R, 'baselines', f'{case}.json')))
        BM = np.load(os.path.join(R, 'baselines', f'{case}_marg.npz'))
        b2p = os.path.join(R, 'baselines2', f'{case}.json')
        base2 = json.load(open(b2p)) if os.path.exists(b2p) else None
        BM2 = np.load(os.path.join(R, 'baselines2', f'{case}_marg.npz')) if base2 else None
        nrep = len(est['reps'])
        rows = []
        for b in sorted(gt):
            g = gt[b]
            ws = [r['wcond_at'][str(b)] for r in est['reps']]
            wE = float(np.mean(ws))
            PC = np.mean([EM[f'r{r}_C_{b}'] for r in range(nrep)], 0)
            PD = np.mean([EM[f'r{r}_D_{b}'] for r in range(nrep)], 0)
            PE = wE * PC + (1 - wE) * PD
            row = dict(beta=b, w_gt=g['w'], w_gt_spread=g['w_spread'], w_gt_se=g.get('w_se'), gt_switches=g['switches'],
                       gt_floor_tv=g['floor'],
                       twophase=dict(w=wE, w_sd=float(np.std(ws)), dlogit=logit(wE) - logit(g['w']),
                                     tv=rowtv(PE, g['P'])))
            # single replicate (budget of one estimator run for the whole curve)
            s1 = []
            for r in range(nrep):
                w1 = est['reps'][r]['wcond_at'][str(b)]
                P1 = w1 * EM[f'r{r}_C_{b}'] + (1 - w1) * EM[f'r{r}_D_{b}']
                s1.append((abs(w1 - g['w']), rowtv(P1, g['P'])))
            row['twophase_single'] = dict(abs_dw_mean=float(np.mean([x[0] for x in s1])),
                                          abs_dw_max=float(np.max([x[0] for x in s1])),
                                          tv_mean=float(np.mean([x[1] for x in s1])),
                                          tv_max=float(np.max([x[1] for x in s1])))
            row['twophase_reps'] = [dict(dw=abs(est['reps'][r]['wcond_at'][str(b)] - g['w']),
                                         tv=rowtv(est['reps'][r]['wcond_at'][str(b)] * EM[f'r{r}_C_{b}'] +
                                                  (1 - est['reps'][r]['wcond_at'][str(b)]) * EM[f'r{r}_D_{b}'], g['P']))
                                    for r in range(nrep)]
            for mname, v in base['betas'][str(b)].items():
                row[mname] = dict(w=v['wC_est'], dlogit=logit(v['wC_est']) - logit(g['w']),
                                  tv=rowtv(BM[f'{b}_{mname}'], g['P']))
            if base2:
                for mname, v in base2['methods'].items():
                    if str(b) not in v['w']:
                        continue
                    wv = v['w'][str(b)]
                    row[mname] = dict(w=wv, dlogit=logit(wv) - logit(g['w']), tv=rowtv(BM2[f'{b}_{mname}'], g['P']))
            rows.append(row)
        summ = {}
        summ['twophase_single_run'] = dict(
            max_dw=float(np.mean([max(r['twophase_reps'][i]['dw'] for r in rows) for i in range(nrep)])),
            max_tv=float(np.mean([max(r['twophase_reps'][i]['tv'] for r in rows) for i in range(nrep)])),
            max_dw_worst=float(max(max(r['twophase_reps'][i]['dw'] for r in rows) for i in range(nrep))))
        for m in ('mh_random', 'mh_witness', 'annealed', 'pt', 'pt_tuned', 'wl_score', 'umb_k'):
            if all(m in r for r in rows):
                summ[m] = dict(max_dw=float(max(abs(r[m]['w'] - r['w_gt']) for r in rows)),
                               max_tv=float(max(r[m]['tv'] for r in rows)))
        summ['gt_floor_max'] = float(max(r['gt_floor_tv'] for r in rows))
        summ['gt_w_se_max'] = float(max((r['w_gt_se'] or 0) for r in rows))
        summ['n_beta'] = len(rows)
        summ['budget_2p'] = est['reps'][0].get('proposals')
        summ['budget_local_per_beta_log2'] = base['logB']
        summ['budget_strong_total_log2'] = base2['logB'] if base2 else None
        summ['gt_floor_mean'] = float(np.mean([r['gt_floor_tv'] for r in rows]))
        allres[case] = dict(source=src, n=est['n'], beta_c_est=[r['beta_c'] for r in est['reps']], summary=summ,
                            proposals_twophase=est['reps'][0].get('proposals'),
                            proposals_baseline_per_beta=2 ** base['logB'], rows=rows)
        # beta_c from GT: interpolate logit(w_gt) = 0
        bs = np.array(sorted(gt))
        lg = np.array([logit(gt[b]['w']) for b in bs])
        i = np.where(np.diff(np.sign(lg)) != 0)[0]
        allres[case]['beta_c_gt'] = (float(bs[i[0]] - lg[i[0]] * (bs[i[0] + 1] - bs[i[0]]) / (lg[i[0] + 1] - lg[i[0]]))
                                     if len(i) else None)
        print(f"\n== {case} (n={est['n']}, GT={src}); beta_c est {np.mean(allres[case]['beta_c_est']):.4f}"
              f" ± {np.std(allres[case]['beta_c_est']):.4f}, GT {allres[case]['beta_c_gt']}")
        print('beta   w_gt  (sw)    | 2ph w   tv    | mhR w  tv    | mhW w  tv    | ann w  tv    | pt w   tv   | floor')
        for r in rows:
            s = f"{r['beta']:.3f} {r['w_gt']:.3f} ({min(r['gt_switches'])}) | "
            for m in ('twophase', 'mh_random', 'mh_witness', 'annealed', 'pt'):
                s += f"{r[m]['w']:.3f} {r[m]['tv']:.3f} | "
            s += f"{r['gt_floor_tv']:.3f}"
            print(s)
    json.dump(allres, open(os.path.join(R, 'gt_comparison.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
