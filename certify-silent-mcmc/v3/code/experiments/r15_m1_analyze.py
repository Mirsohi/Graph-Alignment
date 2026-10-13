"""R15 M1 analysis: regenerates every M1 number from results/r15/fpti_cert/*.json.
Writes results/r15/summary_fpti_cert.json and paper/tables/r15_fpti_cert.tex (draft).
RIGOROUS quantities (deterministic floors) and TI-BASED context are kept in separate keys.
Usage: python r15_m1_analyze.py"""
import glob, json, os
import numpy as np
from r15_common import RES, V3, Enc

D = os.path.join(RES, 'fpti_cert')
REL = json.load(open(os.path.join(V3, 'results', 'summary_released_22.json')))['pairs']
RETURNED = sorted(c for c, v in REL.items() if v['final'] in ('pass', 'pass (retry)'))


def load():
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, '*.json'))) if 'jobs_done' not in f]
    return [r for r in recs if 'RIGOROUS' in r]


def ev(r, b):
    return next(e for e in r['RIGOROUS']['eval'] if abs(e['beta'] - b) < 1e-6)


def med(x):
    return float(np.median(x)) if len(x) else None


PRIMARY_SEEDS = {21: (101, 102), 19: (101,), 17: (101, 102)}   # pre-registered; X2 adds seeds 103,104 at 2^21


def protein(recs, x2=False):
    P = [r for r in recs if r['info']['kind'] == 'protein'
         and (x2 or r['seed'] in PRIMARY_SEEDS[r['logT_per_point']])]
    if x2:
        x2cases = sorted(set(r['case'] for r in P if r['seed'] in (103, 104)))
        P = [r for r in P if r['case'] in x2cases and r['logT_per_point'] == 21]
    out = {}
    for logT in sorted(set(r['logT_per_point'] for r in P), reverse=True):
        R = [r for r in P if r['logT_per_point'] == logT]
        cases = sorted(set(r['case'] for r in R))
        seeds = sorted(set(r['seed'] for r in R))
        per = {}
        for c in cases:
            rs = sorted([r for r in R if r['case'] == c], key=lambda r: r['seed'])
            bc = float(np.mean(rs[0]['info']['beta_c_2P']))
            per[c] = dict(n=rs[0]['n'], seeds=[r['seed'] for r in rs], beta_c_2P_TI_BASED=bc,
                          released_status=REL[c]['final'],
                          E3=[ev(r, 3.0)['E_cert_point'] for r in rs], E3_ball=[ev(r, 3.0)['E_cert_ball'] for r in rs],
                          E25=[ev(r, 2.5)['E_cert_point'] for r in rs], E20=[ev(r, 2.0)['E_cert_point'] for r in rs],
                          first_cert=[r['RIGOROUS']['first_cert_beta_point'] for r in rs],
                          first_cert_ball=[r['RIGOROUS']['first_cert_beta_ball'] for r in rs],
                          nucleation=[r['nucleation_beta'] for r in rs],
                          S_star_source=[r['S_star_source'] for r in rs], S_star=[r['S_star'] for r in rs],
                          est_err3_TI_BASED=[float(np.mean([x[-1] for x in r['TI_BASED_context']['est_err']])) for r in rs],
                          proposals=[r['proposals_ti'] for r in rs], seconds=[r['prov']['wall_seconds'] for r in rs])
            p = per[c]
            p['certified_all_seeds_3'] = bool(all(e > 0 for e in p['E3']))
            p['certified_any_seed_3'] = bool(any(e > 0 for e in p['E3']))
            p['certified_all_seeds_3_ball'] = bool(all(e > 0 for e in p['E3_ball']))
            p['nuc_offset'] = [(x - bc) if x is not None else None for x in p['nucleation']]
        full = [c for c in cases if len(per[c]['seeds']) == len(seeds)]
        allE3 = [e for c in cases for e in per[c]['E3']]
        minE3 = [min(per[c]['E3']) for c in full]
        nuc = [o for c in cases for o in per[c]['nuc_offset'] if o is not None]
        cnt = lambda key, sub: sum(per[c][key] for c in sub)
        out[f'T{logT}'] = dict(
            logT_per_point=logT, seeds=seeds, n_pairs=len(cases), n_pairs_all_seeds=len(full),
            log2_total_proposals_per_run=float(np.log2(R[0]['proposals_ti'])),
            RIGOROUS=dict(
                certified_all_seeds_at3=cnt('certified_all_seeds_3', full), denominator=len(full),
                certified_any_seed_at3=cnt('certified_any_seed_3', full),
                runs_certified_at3=int(sum(e > 0 for e in allE3)), runs=len(allE3),
                certified_all_seeds_at3_ball=cnt('certified_all_seeds_3_ball', full),
                median_E3_runs=med(allE3), max_E3_runs=float(max(allE3)) if allE3 else None,
                min_E3_runs=float(min(allE3)) if allE3 else None,
                median_minseed_E3=med(minE3), max_minseed_E3=float(max(minE3)) if minE3 else None,
                runs_certified_at25=int(sum(e > 0 for c in cases for e in per[c]['E25'])),
                runs_certified_at20=int(sum(e > 0 for c in cases for e in per[c]['E20'])),
                not_returned_16=dict(certified_all_seeds=cnt('certified_all_seeds_3', [c for c in full if c not in RETURNED]),
                                     denominator=len([c for c in full if c not in RETURNED])),
                returned_6=dict(certified_all_seeds=cnt('certified_all_seeds_3', [c for c in full if c in RETURNED]),
                                denominator=len([c for c in full if c in RETURNED])),
                not_certified_pairs=[c for c in full if not per[c]['certified_all_seeds_3']]),
            median_nucleation_offset_TI_BASED_betac=med(nuc),
            TI_BASED=dict(note='two-phase estimated error Lambda_2P(3)-Lambda_FP(3); context only',
                          median_est_err3=med([e for c in cases for e in per[c]['est_err3_TI_BASED']])),
            compute=dict(total_proposals=int(sum(r['proposals_ti'] for r in R)),
                         total_seconds=float(sum(r['prov']['wall_seconds'] for r in R))),
            pairs=per)
    return out


def silent(recs):
    S = [r for r in recs if r['info']['kind'] == 'silent']
    out = {}
    for logT in sorted(set(r['logT_per_point'] for r in S), reverse=True):
        R = [r for r in S if r['logT_per_point'] == logT]
        tags = sorted(set(r['info']['tag'] for r in R))
        per = {}
        for t in tags:
            rs = sorted([r for r in R if r['info']['tag'] == t], key=lambda r: r['seed'])
            per[t] = dict(witness_success=rs[0]['info']['witness_success'], seeds=[r['seed'] for r in rs],
                          beta_c_ref=rs[0]['info']['beta_c_ref'],
                          cert=[[e['E_cert_point'] > 0 for e in r['RIGOROUS']['eval']] for r in rs],
                          E=[[e['E_cert_point'] for e in r['RIGOROUS']['eval']] for r in rs],
                          cert_ball=[[e['E_cert_ball'] > 0 for e in r['RIGOROUS']['eval']] for r in rs],
                          S_star_source=[r['S_star_source'] for r in rs])
        offs = ['-0.1', '+0.03', '+0.08', '+0.15']
        grp = {}
        for name, sel in (('witness_success', True), ('witness_failure', False)):
            ts = [t for t in tags if per[t]['witness_success'] == sel]
            g = dict(n_instances=len(ts), offsets=offs)
            g['instances_certified_all_seeds'] = [sum(all(c[j] for c in per[t]['cert']) for t in ts) for j in range(4)]
            g['instances_certified_any_seed'] = [sum(any(c[j] for c in per[t]['cert']) for t in ts) for j in range(4)]
            g['runs_certified'] = [sum(c[j] for t in ts for c in per[t]['cert']) for j in range(4)]
            g['runs_certified_ball'] = [sum(c[j] for t in ts for c in per[t]['cert_ball']) for j in range(4)]
            g['runs'] = sum(len(per[t]['cert']) for t in ts)
            grp[name] = g
        out[f'T{logT}'] = dict(RIGOROUS=grp, instances=per,
                               compute=dict(total_proposals=int(sum(r['proposals_ti'] for r in R)),
                                            witness_search_proposals=int(sum(r['info']['witness_search_proposals'] for r in R)),
                                            total_seconds=float(sum(r['prov']['wall_seconds'] for r in R))))
    return out


def table(summ):
    L = [r'% R15 M1 draft (r15_m1_analyze.py). RIGOROUS: witness floor beta*S* vs standard TI along beta.',
         r'% Nucleation offset uses the TI-based two-phase beta_c (context).',
         r'\begin{tabular}{lcccccc}', r'\toprule',
         r'TI budget & seeds & certified at $\beta{=}3$ & certified among & median ${\rm Err}_{\rm cert}(3)$ & max ${\rm Err}_{\rm cert}(3)$ & median $\beta_{\rm nuc}-\hat\beta_c$ \\',
         r'(per point / total) & & (all seeds) & not returned (all seeds) & (nats, rigorous) & (nats, rigorous) & (TI-based $\hat\beta_c$) \\',
         r'\midrule']
    for k, v in summ['protein'].items():
        R = v['RIGOROUS']
        L.append(f"$2^{{{v['logT_per_point']}}}$ / $2^{{{v['log2_total_proposals_per_run']:.1f}}}$ & {len(v['seeds'])} & "
                 f"{R['certified_all_seeds_at3']}/{R['denominator']} & {R['not_returned_16']['certified_all_seeds']}/{R['not_returned_16']['denominator']} & "
                 f"{R['median_E3_runs']:.1f} & {R['max_E3_runs']:.1f} & {v['median_nucleation_offset_TI_BASED_betac']:+.2f} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(L) + '\n'


def silent_table(summ):
    L = [r'% R15 M1 draft: silent synthetic pairs, rigorous witness floor vs standard TI (runs certified / runs).',
         r'\begin{tabular}{lccccc}', r'\toprule',
         r'TI budget & instances & $\beta_c^{\rm ref}-0.1$ & $+0.03$ & $+0.08$ & $+0.15$ \\', r'\midrule']
    for k, v in summ['silent'].items():
        for name in ('witness_success', 'witness_failure'):
            g = v['RIGOROUS'][name]
            L.append(f"$2^{{{k[1:]}}}$ & {name.replace('_', ' ')} ({g['n_instances']}) & " +
                     ' & '.join(f"{x}/{g['runs']}" for x in g['runs_certified']) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(L) + '\n'


def main():
    recs = load()
    summ = dict(note='RIGOROUS = deterministic witness floor; TI_BASED = two-phase context. Every count states its denominator.',
                returned_pairs=RETURNED, protein=protein(recs), silent=silent(recs),
                X2_extra_seeds=dict(note='STRETCH X2: 2^21 with seeds 101-104 on the released-pass pairs; not the pre-registered primary count',
                                    protein=protein(recs, x2=True)))
    summ['compute_total'] = dict(proposals=int(sum(r['proposals_ti'] for r in recs)),
                                 seconds=float(sum(r['prov']['wall_seconds'] for r in recs)), runs=len(recs))
    json.dump(summ, open(os.path.join(RES, 'summary_fpti_cert.json'), 'w'), cls=Enc, indent=1)
    T = os.path.join(V3, 'paper', 'tables')
    if summ['protein']:
        open(os.path.join(T, 'r15_fpti_cert.tex'), 'w').write(table(summ))
    if summ['silent']:
        open(os.path.join(T, 'r15_fpti_silent.tex'), 'w').write(silent_table(summ))
    return summ


if __name__ == '__main__':
    s = main()
    for k, v in s['protein'].items():
        print(k, json.dumps({x: y for x, y in v['RIGOROUS'].items()}, cls=Enc))
    for k, v in s['silent'].items():
        print('silent', k, json.dumps(v['RIGOROUS'], cls=Enc))
