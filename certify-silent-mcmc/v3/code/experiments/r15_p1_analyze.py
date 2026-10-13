"""R15 M2 analysis (FROZEN with PREREG.json; presentation lives in r15_p1_present.py).

Unit = pipeline run (instance x replicate).  Error per run e = max_{beta in G} |a_hat(beta) - a_ref(beta)|.
Operating points:
  O0    : return a_hat whenever a beta_c crossing exists (no screen);
  O1    : released screen, first two-seed attempt passes ('pass');
  O2    : PRIMARY, released screen including the single retry ('pass' or 'pass (retry)');
  O2(t) : as O2 with window tolerance |logit w_window - logit w_hat| < t, t in {0.25, 0.5, 1, 2}, recomputed
          from the stored per-seed rule values (no rerun); other rules unchanged.  Attempt status: 'pass' if
          both seeds pass every rule; 'multi-phase' if the middle-basin rule fires in both seeds; else
          'unresolved'.  If the first attempt is 'unresolved' under t, the stored retry attempt is used; if
          the released procedure (t = 1) did not run a retry, no retry data exist and the run ABSTAINS
          (conservative for coverage).  O2(1) must reproduce O2 exactly (asserted).
Comparators return always (coverage 1); unit = instance.
Metrics at eps in {0.05 (primary), 0.10}: coverage = returned / all runs (no-crossing and unresolved stay in the
denominator); false-pass rate (FPR) = #(returned, e > eps) / #returned; e quantiles (median, 90%, max;
numpy 'linear') over returned runs, and for O0 also over ALL runs (no-crossing runs scored with their
unscreened a_hat); cost per useful output = total proposals (and seconds) over all runs / #(returned with
e <= eps); all per family as well.  95% intervals: instance-cluster bootstrap, 2000 resamples, seed 0.
Pre-registered criteria:
  S-a: at O2, eps = 0.05: FPR <= 0.10 AND FPR <= 0.5 * FPR(O0) on the same runs; AND O2 coverage on F1 u F3 >= 0.5.
  S-b: over runs returned at O2, the pipeline's 90% error <= the better comparator's 90% error, where each
       returned pipeline run is paired with its instance's comparator error (multiset over the same runs).
Secondary: mean row-TV of marginals at G vs the reference (average of the two reference seeds' marginals),
with the reference noise floor (row-TV between the two reference seeds).
Usage: python r15_p1_analyze.py [p1_dir]  -> <p1_dir>/summary_p1.json"""
import json, os, sys, glob
import numpy as np
from r15_common import Enc
from r15_p1_common import P1, G, TEST, row_tv

TOLS = (0.25, 0.5, 1.0, 2.0)
EPS = (0.05, 0.10)
METHODS = ('wl_ext', 'pt_broad_w')


def lg(x):
    return np.log(max(x, 1e-300) / max(1 - x, 1e-300))


def attempt_status(att, t, w_hat):
    runs = []
    for r in att['runs']:
        rules = dict(r['rules'])
        rules['window'] = bool(abs(lg(r['w_window']) - lg(w_hat)) < t)
        runs.append(rules)
    if all(all(x.values()) for x in runs):
        return 'pass'
    if all(not x['middle_basin'] for x in runs):
        return 'multi-phase'
    return 'unresolved'


def returned_at(rec, t):
    """O2(t) decision from stored values."""
    sc = rec['screen']
    if sc['status'] == 'no-crossing':
        return False, 'no-crossing'
    atts = sc['attempts']
    s1 = attempt_status(atts[0], t, sc['w_hat_used'])
    if s1 != 'unresolved':
        return s1 == 'pass', s1
    if len(atts) < 2:
        return False, 'unresolved (no retry data)'
    s2 = attempt_status(atts[1], t, sc['w_hat_used'])
    return s2 == 'pass', ('pass (retry)' if s2 == 'pass' else s2)


def q(x, p):
    return float(np.quantile(x, p)) if len(x) else None


def block(runs, sel, eps, key='e'):
    """metrics over `runs` (list of dicts) with returned mask sel."""
    N = len(runs)
    ret = [r for r, s in zip(runs, sel) if s]
    e = np.array([r[key] for r in ret])
    out = dict(n_runs=N, n_returned=len(ret), coverage=len(ret) / N if N else None)
    for ep in eps:
        bad = int((e > ep).sum())
        good = len(ret) - bad
        out[f'eps{ep}'] = dict(false_pass=bad, fpr=(bad / len(ret) if ret else None), useful=good,
                               cost_per_useful=(sum(r['cost'] for r in runs) / good if good else None),
                               sec_per_useful=(sum(r['sec'] for r in runs) / good if good else None))
    out['e_returned'] = dict(median=q(e, .5), q90=q(e, .9), max=(float(e.max()) if len(e) else None))
    out['total_cost'] = int(sum(r['cost'] for r in runs))
    out['total_sec'] = float(sum(r['sec'] for r in runs))
    return out


def bootstrap(insts, fn, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        samp = [insts[i] for i in rng.integers(len(insts), size=len(insts))]
        v = fn(samp)
        if v is not None:
            vals.append(v)
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if vals else None


def main(p1=P1, insts=None):
    insts = insts or TEST
    ref, pipe, comp = {}, {}, {}
    for i in insts:
        f = os.path.join(p1, 'ref', i + '.json')
        if os.path.exists(f):
            ref[i] = json.load(open(f))
        for r in (1, 2):
            f = os.path.join(p1, 'pipe', f'{i}_r{r}.json')
            if os.path.exists(f):
                pipe[(i, r)] = json.load(open(f))
        for m in METHODS:
            f = os.path.join(p1, 'comp', f'{i}_{m}.json')
            if os.path.exists(f):
                comp[(i, m)] = json.load(open(f))
    complete = [i for i in insts if i in ref and (i, 1) in pipe and (i, 2) in pipe
                and all((i, m) in comp for m in METHODS)]
    pipe_complete = [i for i in insts if i in ref and (i, 1) in pipe and (i, 2) in pipe]
    runs = []
    for i in pipe_complete:
        aref = np.array([r['a_ref'] for r in ref[i]['rows']])
        Rm = np.load(os.path.join(p1, 'ref', i + '_marg.npz'))
        for rep in (1, 2):
            rec = pipe[(i, rep)]
            ah = np.array([e['a_hat'] for e in rec['estimate']])
            Pm = np.load(os.path.join(p1, 'pipe', f'{i}_r{rep}_marg.npz'))
            tv = []
            for e in rec['estimate']:
                b = e['beta']
                M = e['wC'] * Pm[f'{b}_C'] + (1 - e['wC']) * Pm[f'{b}_D']
                Mr = 0.5 * (Rm[f'{b}_s9001'] + Rm[f'{b}_s9002'])
                tv.append(row_tv(M, Mr))
            st = rec['screen']['status']
            d = dict(inst=i, fam=i[:2], rep=rep, e=float(np.max(np.abs(ah - aref))), err_beta=(ah - aref).tolist(),
                     rowtv=float(np.mean(tv)), status=st, crossing=rec['crossing'],
                     O0=bool(rec['crossing']), O1=bool(rec['screen']['attempts'][0]['status'] == 'pass') if rec['crossing'] else False,
                     O2=st in ('pass', 'pass (retry)'), cost=rec['cost_total'], cost_parts=rec['cost'],
                     sec=rec['prov']['wall_seconds'], ref_uncertain=ref[i]['reference_uncertain'])
            for t in TOLS:
                d[f'O2_t{t}'], d[f'O2_t{t}_status'] = returned_at(rec, t)
            assert d['O2_t1.0'] == d['O2'], (i, rep)
            runs.append(d)
    ops = ['O0', 'O1', 'O2'] + [f'O2_t{t}' for t in TOLS]
    fams = sorted(set(r['fam'] for r in runs))
    res = dict(G=G, n_instances_pipeline=len(pipe_complete), n_instances_complete=len(complete),
               instances_pipeline=pipe_complete, instances_complete=complete, runs=runs, ops={}, per_family={},
               comparators={}, criteria={})
    by_inst = lambda rs: {i: [r for r in rs if r['inst'] == i] for i in pipe_complete}
    for op in ops:
        sel = [r[op] for r in runs]
        b = block(runs, sel, EPS)
        if op == 'O0':
            ea = np.array([r['e'] for r in runs])
            b['e_all_runs'] = dict(median=q(ea, .5), q90=q(ea, .9), max=float(ea.max()) if len(ea) else None)
        groups = by_inst(runs)

        def fpr_fn(samp, op=op):
            rs = [r for i in samp for r in groups[i]]
            ret = [r for r in rs if r[op]]
            return (sum(r['e'] > 0.05 for r in ret) / len(ret)) if ret else None

        def cov_fn(samp, op=op):
            rs = [r for i in samp for r in groups[i]]
            return sum(r[op] for r in rs) / len(rs)

        def q90_fn(samp, op=op):
            rs = [r for i in samp for r in groups[i]]
            e = [r['e'] for r in rs if r[op]]
            return q(e, .9)
        if pipe_complete:
            b['ci95'] = dict(fpr_eps005=bootstrap(pipe_complete, fpr_fn), coverage=bootstrap(pipe_complete, cov_fn),
                             q90=bootstrap(pipe_complete, q90_fn))
        res['ops'][op] = b
        for f in fams:
            rf = [r for r in runs if r['fam'] == f]
            res['per_family'].setdefault(f, {})[op] = block(rf, [r[op] for r in rf], EPS)
        rows_rtv = [r['rowtv'] for r in runs if r[op]]
        b['rowtv_returned_median'] = q(rows_rtv, .5)
    # comparators (unit = instance), on the instances with complete comparator runs
    for m in METHODS:
        cr = []
        for i in complete:
            c = comp[(i, m)]
            aref = np.array([r['a_ref'] for r in ref[i]['rows']])
            a = np.array([r['a'] for r in c['rows']])
            Rm = np.load(os.path.join(p1, 'ref', i + '_marg.npz'))
            Cm = np.load(os.path.join(p1, 'comp', f'{i}_{m}_marg.npz'))
            tv = np.mean([row_tv(Cm[f'{b}'], 0.5 * (Rm[f'{b}_s9001'] + Rm[f'{b}_s9002'])) for b in G])
            cr.append(dict(inst=i, fam=i[:2], e=float(np.max(np.abs(a - aref))), err_beta=(a - aref).tolist(),
                           rowtv=float(tv), cost=c['charged_proposals'], budget=c['budget_total'],
                           sec=c['prov']['wall_seconds']))
        blk = block(cr, [True] * len(cr), EPS)
        blk['per_family'] = {f: block([r for r in cr if r['fam'] == f], [True] * sum(r['fam'] == f for r in cr), EPS)
                             for f in fams}
        blk['instances'] = cr
        blk['rowtv_median'] = q([r['rowtv'] for r in cr], .5)
        res['comparators'][m] = blk
    # reference noise floor
    res['reference'] = dict(uncertain=[i for i in pipe_complete if ref[i]['reference_uncertain']],
                            max_se_ref={i: ref[i]['max_se_ref'] for i in pipe_complete},
                            rowtv_floor_median=q([np.mean([r['rowtv_between_seeds'] for r in ref[i]['rows']])
                                                  for i in pipe_complete], .5))
    # criteria
    O0, O2 = res['ops'].get('O0'), res['ops'].get('O2')
    if runs:
        f2, f0 = O2['eps0.05']['fpr'], O0['eps0.05']['fpr']
        ctrl = [r for r in runs if r['fam'] in ('F1', 'F3')]
        cov_ctrl = sum(r['O2'] for r in ctrl) / len(ctrl) if ctrl else None
        Sa = dict(fpr_O2=f2, fpr_O0=f0, coverage_O2_controls=cov_ctrl,
                  cond1=(f2 is not None and f2 <= 0.10), cond2=(f2 is not None and f0 is not None and f2 <= 0.5 * f0),
                  cond3=(cov_ctrl is not None and cov_ctrl >= 0.5))
        Sa['pass'] = bool(Sa['cond1'] and Sa['cond2'] and Sa['cond3'])
        res['criteria']['S_a'] = Sa
        ret = [r for r in runs if r['O2'] and r['inst'] in complete]
        if ret and complete:
            pq = q([r['e'] for r in ret], .9)
            cq = {m: q([next(c['e'] for c in res['comparators'][m]['instances'] if c['inst'] == r['inst']) for r in ret], .9)
                  for m in METHODS}
            best = min(cq.values())
            res['criteria']['S_b'] = dict(pipeline_q90=pq, comparator_q90=cq, better_comparator=min(cq, key=cq.get),
                                          n_runs=len(ret), n_instances=len(set(r['inst'] for r in ret)),
                                          pass_=bool(pq <= best))
    res['main_text_eligible'] = len(complete) >= 24
    out = os.path.join(p1, 'summary_p1.json')
    tmp = out + '.tmp'
    json.dump(res, open(tmp, 'w'), cls=Enc, indent=1)
    os.replace(tmp, out)
    return res


if __name__ == '__main__':
    p1 = sys.argv[1] if len(sys.argv) > 1 else P1
    insts = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    r = main(p1, insts)
    print(json.dumps(dict(n_pipe=r['n_instances_pipeline'], n_complete=r['n_instances_complete'],
                          criteria=r['criteria'],
                          ops={k: (v['coverage'], v['eps0.05']['fpr'], v['e_returned']['q90']) for k, v in r['ops'].items()},
                          comps={k: (v['eps0.05']['fpr'], v['e_returned']['q90']) for k, v in r['comparators'].items()}),
                     cls=Enc, indent=1))
