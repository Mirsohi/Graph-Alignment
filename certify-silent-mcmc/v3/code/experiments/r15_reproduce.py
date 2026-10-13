"""R15: regenerate every R15 summary, draft table and draft figure from saved results (no sampler re-runs).
  M1  r15_m1_analyze.py         -> results/r15/summary_fpti_cert.json, paper/tables/r15_fpti_cert.tex, r15_fpti_silent.tex
  M2  r15_p1_analyze.py (frozen) -> results/r15/p1_heldout/summary_p1.json
      r15_p1_present.py         -> paper/tables/r15_p1.tex, r15_p1_family.tex, figures/r15_p1_accuracy_coverage.pdf
  S1  r15_mass_floor_analyze.py -> results/r15/summary_mass_floor.json, paper/tables/r15_mass_floor.tex
  S2  r15_delay_check.py        -> results/r15/summary_delay.json
  S3  r15_pt_ti_analyze.py      -> results/r15/summary_pt_ti.json, paper/tables/r15_pt_ti.tex
  S4  r15_fig_takeaway.py       -> figures/r15_takeaway.pdf (main-text Figure 2)
  X1  r15_x1_analyze.py         -> results/r15/p1_heldout/summary_x1.json
  paper r15_paper_tables.py     -> paper/tables/r15_p1_main.tex (main text), r15_p1_subsets.tex, r15_x1.tex,
                                   r15_pairs.tex, r15_x2.tex, r15_ledger.tex (appendix)
  paper fig_gt.py --top         -> figures/gt_phase_top.pdf (main-text Figure 3; reads results/gt_comparison.json)
Usage: python r15_reproduce.py"""
import os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for s, args in (('r15_m1_analyze.py', []), ('r15_p1_analyze.py', []), ('r15_p1_present.py', []), ('r15_p1_subsets.py', []),
                ('r15_mass_floor_analyze.py', []), ('r15_delay_check.py', []), ('r15_pt_ti_analyze.py', []),
                ('r15_x1_analyze.py', []), ('r15_fig_takeaway.py', ['--coset']), ('r15_paper_tables.py', []),
                ('fig_gt.py', ['--top'])):
    print(f'== R15 {s}', flush=True)
    sys.argv = [s] + args
    try:
        runpy.run_path(os.path.join(HERE, s), run_name='__main__')
    except Exception as e:      # an item whose results are absent is reported, not fatal
        print(f'   skipped: {e!r}')
