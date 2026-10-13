"""R15 (paper revision): LaTeX tables for the main text and appendix, generated from the saved R15
summaries and per-job result files only (no sampler is re-run).  Called by r15_reproduce.py.

  paper/tables/r15_p1_main.tex     main-text accuracy-coverage table (EMPIRICAL; summary_p1.json)
  paper/tables/r15_p1_subsets.tex  reference-certain/uncertain and control/hidden-middle subsets
                                   (summary_p1_subsets.json)
  paper/tables/r15_x1.tex          comparators with a second seed and an agreement abstention (summary_x1.json)
  paper/tables/r15_pairs.tex       per-pair RIGOROUS certified deficits at beta=3: standard TI (2^21, 2^17),
                                   coset-ELBO floor, PT-based TI (summary_fpti_cert.json, summary_mass_floor.json,
                                   summary_pt_ti.json)
  paper/tables/r15_x2.tex          extra standard-TI seeds on the six released-pass pairs (summary_fpti_cert.json)
  paper/tables/r15_ledger.tex      compute ledger summed over the per-job result files (results/r15/**)

Usage: python r15_paper_tables.py"""
import glob, json, math, os
from r15_common import V3, RES

T = os.path.join(V3, 'paper', 'tables')
P1 = os.path.join(RES, 'p1_heldout')


def name(c):
    return {'adk_crystal_control': 'AdK cryst.', 'adk_open_closed': 'AdK o/c'}.get(c, c.replace('pscdb_', '').replace('_', '.'))


def f2(x):
    return '--' if x is None else f'{x:.2f}'


def frac(a, b):
    return f'{a}/{b}'


def write(fn, lines):
    with open(os.path.join(T, fn), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def p1_main():
    S = json.load(open(os.path.join(P1, 'summary_p1.json')))
    o, c = S['ops'], S['comparators']
    rows = [('no screen', o['O0']), ('released screen (primary)', o['O2']),
            ('stricter window, $t{=}0.5$', o['O2_t0.5']), ('stricter window, $t{=}0.25$', o['O2_t0.25'])]
    L = ['% R15 M2 (r15_paper_tables.py from results/r15/p1_heldout/summary_p1.json). EMPIRICAL; synthetic held-out pairs.',
         '\\begin{tabular}{lccc}', '\\toprule',
         'operating point & returned & false passes & $e$, 90\\% \\\\', '\\midrule']
    for lab, d in rows:
        L.append(f"{lab} & {frac(d['n_returned'], d['n_runs'])} & {frac(d['eps0.05']['false_pass'], d['n_returned'])} & "
                 f"{d['e_returned']['q90']:.2f} \\\\")
    L.append('\\midrule')
    for m, lab in (('wl_ext', 'Wang--Landau, matched'), ('pt_broad_w', 'witness PT, matched')):
        d = c[m]
        L.append(f"{lab} & {frac(d['n_returned'], d['n_runs'])} & {frac(d['eps0.05']['false_pass'], d['n_returned'])} & "
                 f"{d['e_returned']['q90']:.2f} \\\\")
    L += ['\\bottomrule', '\\end{tabular}']
    write('r15_p1_main.tex', L)


def p1_subsets():
    S = json.load(open(os.path.join(P1, 'summary_p1_subsets.json')))
    L = ['% R15 M2 subsets (r15_paper_tables.py from summary_p1_subsets.json). EMPIRICAL.',
         '\\begin{tabular}{lcccccc}', '\\toprule',
         'subset & instances & runs & O0 false passes & O2 returned & O2 false passes & O2 $e$, 90\\% \\\\', '\\midrule']
    for k, lab in (('reference_certain', 'reference-certain'), ('reference_uncertain', 'reference-uncertain'),
                   ('controls_F1_F3', 'controls (F1, F3)'), ('hidden_F2_F4', 'hidden middle (F2, F4)')):
        d = S[k]
        o0, o2 = d['ops']['O0'], d['ops']['O2']
        q = o2['e_returned']['q90'] if o2['n_returned'] else None
        L.append(f"{lab} & {d['n_instances']} & {d['n_runs']} & {frac(o0['eps0.05']['false_pass'], o0['n_returned'])} & "
                 f"{frac(o2['n_returned'], o2['n_runs'])} & {frac(o2['eps0.05']['false_pass'], o2['n_returned'])} & {f2(q)} \\\\")
    L += ['\\bottomrule', '\\end{tabular}']
    write('r15_p1_subsets.tex', L)


def x1():
    S = json.load(open(os.path.join(P1, 'summary_p1.json')))['comparators']
    X = json.load(open(os.path.join(P1, 'summary_x1.json')))['methods']
    L = ['% R15 X1 (r15_paper_tables.py from summary_p1.json and summary_x1.json). EMPIRICAL.',
         '\\begin{tabular}{llcccc}', '\\toprule',
         'comparator & rule & budget & returned & false passes & $e$, 90\\% \\\\', '\\midrule']
    for m, lab in (('wl_ext', 'Wang--Landau'), ('pt_broad_w', 'witness PT')):
        a = S[m]
        L.append(f"{lab} & one seed & $B$ & {frac(a['n_returned'], a['n_runs'])} & {frac(a['eps0.05']['false_pass'], a['n_returned'])} & "
                 f"{a['e_returned']['q90']:.2f} \\\\")
        b = X[m]['summary']
        L.append(f" & two seeds agree & $2B$ & {frac(b['n_returned'], b['n_runs'])} & {frac(b['eps0.05']['false_pass'], b['n_returned'])} & "
                 f"{b['e_returned']['q90']:.2f} \\\\")
    L += ['\\bottomrule', '\\end{tabular}']
    write('r15_x1.tex', L)


def pairs():
    F = json.load(open(os.path.join(RES, 'summary_fpti_cert.json')))['protein']
    M = json.load(open(os.path.join(RES, 'summary_mass_floor.json')))['protein']
    PT = json.load(open(os.path.join(RES, 'summary_pt_ti.json')))['pairs']
    coset = {}
    for r in M['runs']:
        coset[(r['case'], r['logT'], r['seed'])] = r['E3']['coset']
    p21, p17 = F['T21']['pairs'], F['T17']['pairs']
    order = sorted(p21, key=lambda c: (p21[c]['n'], c))

    def pr(v):
        return '/'.join(f'${x:+.1f}$' for x in v)

    L = ['% R15 per-pair RIGOROUS certified deficits E_cert(3) = floor - Lambda_hat(3), nats (r15_paper_tables.py).',
         '% Positive = certified. Standard TI seeds 101/102; PT-TI seed 1. Status: released two-phase status (heuristic screen).',
         '\\begin{tabular}{lrlcccc}', '\\toprule',
         ' & & released & \\multicolumn{2}{c}{standard TI, $2^{21}$/point} & standard TI, $2^{17}$ & PT-based TI \\\\',
         'pair & $n$ & status & point floor & coset floor & point floor & point floor \\\\', '\\midrule']
    for c in order:
        a = p21[c]
        cs = [coset[(c, 21, s)] for s in a['seeds']]
        st = a['released_status'].replace('multi-phase', 'multi').replace('unresolved', 'unres.')
        L.append(f"{name(c)} & {a['n']} & {st} & {pr(a['E3'])} & {pr(cs)} & {pr(p17[c]['E3'])} & ${PT[c]['E3']:+.1f}$ \\\\")
    L += ['\\bottomrule', '\\end{tabular}']
    write('r15_pairs.tex', L)


def x2():
    X = json.load(open(os.path.join(RES, 'summary_fpti_cert.json')))['X2_extra_seeds']['protein']['T21']
    L = ['% R15 X2 (r15_paper_tables.py from summary_fpti_cert.json -> X2_extra_seeds). RIGOROUS E_cert(3), nats.',
         '\\begin{tabular}{lcc}', '\\toprule', 'pair & seeds certified & ${\\rm Err}_{\\rm cert}(3)$, seeds 101/102/103/104 \\\\', '\\midrule']
    for c, a in sorted(X['pairs'].items(), key=lambda kv: -sum(e > 0 for e in kv[1]['E3'])):
        L.append(f"{name(c)} & {sum(e > 0 for e in a['E3'])}/{len(a['E3'])} & " + ' / '.join(f'${e:+.1f}$' for e in a['E3']) + ' \\\\')
    L += ['\\midrule', f"all & {X['RIGOROUS']['runs_certified_at3']}/{X['RIGOROUS']['runs']} runs & "
          f"{X['RIGOROUS']['certified_all_seeds_at3']}/{X['RIGOROUS']['denominator']} pairs certified on all four seeds \\\\",
          '\\bottomrule', '\\end{tabular}']
    write('r15_x2.tex', L)


def ledger():
    def load(pattern):
        return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, pattern)))
                if not f.endswith(('unit_tests.json',))]

    fp = load('fpti_cert/*.json')
    isx2 = [d.get('info', {}).get('kind') == 'protein' and d['seed'] in (103, 104) for d in fp]
    m1 = [d for d, x in zip(fp, isx2) if not x]
    x2 = [d for d, x in zip(fp, isx2) if x]
    items = [
        ('Standard TI, 22 proteins + 16 synthetic pairs (TI only)', m1, lambda d: d['proposals_ti']),
        ('\\quad synthetic witness re-search (in these runs)', None, sum((d.get('info', {}).get('witness_search_proposals') or 0) for d in m1)),
        ('Held-out references (40 instances)', load('p1_heldout/ref/*.json'), lambda d: d['proposals_approx']),
        ('Held-out pipeline (80 runs, all components)', load('p1_heldout/pipe/*.json'), lambda d: d['cost_total']),
        ('Held-out Wang--Landau (40, charged)', load('p1_heldout/comp/*_wl_ext.json'), lambda d: d['charged_proposals']),
        ('Held-out witness PT (40, charged)', load('p1_heldout/comp/*_pt_broad_w.json'), lambda d: d['charged_proposals']),
        ('Coset-floor blocks (22 + 16 pairs)', [d for d in load('mass_floor/*.json')], lambda d: d.get('chain_proposals', 0)),
        ('PT-based TI (22 pairs)', load('pt_ti/*.json'), lambda d: d.get('proposals', 0)),
        ('Comparators, second seed (80; Table~\\ref{tab:r15x1})', load('p1_heldout/comp_x1/*.json'), lambda d: d['charged_proposals']),
        ('Standard TI, seeds 103--104 (12; Table~\\ref{tab:r15x2})', x2, lambda d: d['proposals_ti']),
    ]
    L = ['% R15 compute ledger (r15_paper_tables.py): sums over the per-job result files; CPU-h = sum of wall-clock seconds / 3600.',
         '\\begin{tabular}{lrrr}', '\\toprule', 'item & jobs & proposals & CPU-h \\\\', '\\midrule']
    TP = TS = 0
    for lab, recs, f in items:
        if recs is None:
            L.append(f'{lab} & & $2^{{{math.log2(f):.2f}}}$ & (incl.) \\\\')
            continue
        P = sum(f(d) for d in recs)
        S = sum(d['prov']['wall_seconds'] for d in recs)
        TP += P
        TS += S
        L.append(f'{lab} & {len(recs)} & $2^{{{math.log2(P):.2f}}}$ & {S / 3600:.2f} \\\\')
    L += ['\\midrule', f'total (excl. smoke tests, post-hoc checks) & & $2^{{{math.log2(TP):.2f}}}$ & {TS / 3600:.1f} \\\\', '\\bottomrule', '\\end{tabular}']
    write('r15_ledger.tex', L)


def main():
    for f in (p1_main, p1_subsets, x1, pairs, x2, ledger):
        f()
    print('wrote r15_p1_main, r15_p1_subsets, r15_x1, r15_pairs, r15_x2, r15_ledger to', T)


if __name__ == '__main__':
    main()
