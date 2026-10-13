# Certifying and Correcting Silent MCMC Failures

Code, data and results for **Certifying and Correcting Silent MCMC Failures: From Variable Selection to Protein Alignment**, Melika Gorgi and Kourosh Mirsohi, 2026.

All commands below are run from this folder (`certify-silent-mcmc/`).

## Quick check

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python code/tests/test_core.py
python v3/code/experiments/reproduce_tables.py
```

The unit tests take about 15 s, and `reproduce_tables.py` regenerates every table from the saved results in about a minute without running a sampler. Both were also checked with Python 3.13 and current releases of NumPy, SciPy, Numba, scikit-learn and Matplotlib: the tests pass and the tables come out byte for byte the same. The tables in the paper use the same numbers; some row and column labels were shortened there for presentation.

## Files left out to keep the repository small

Most per-instance marginal arrays (`*_marg.npz`, about 57 MB) are left out. Everything else is included: all code, data, per-instance JSON results, summaries, tables and figures.

- The 20 marginal arrays that `reproduce_tables.py` reads for Tables 13 and 15–19 (`v3/results/baselines3/`, `v3/results/gt/`) are kept. To save space they are stored as float32 instead of float64. Regenerating from them reproduces every table in `v3/paper/tables/` byte for byte.
- The held-out marginal arrays (`v3/results/r15/p1_heldout/{ref,pipe,comp}/*_marg.npz`) are left out. `r15_p1_analyze.py` therefore reports "skipped", and `r15_p1_present.py` rebuilds the held-out tables from the saved summary in `v3/results/r15/p1_heldout/`. To recompute that summary, run `r15_p1_reference.py`, `r15_p1_pipeline.py` and `r15_p1_comparators.py`, which regenerate the arrays.

## Layout

| Path | Contents |
|---|---|
| `code/gacert/` | Core samplers and diagnostics, Numba-compiled: score differences, Metropolis–Hastings, parallel tempering, split-R̂, IID disagreement floors, power-posterior TI, witness bounds. |
| `code/tests/` | Unit tests and exact enumeration helpers (n ≤ 10). |
| `code/experiments/` | Scripts of an earlier study, used only by `v3/code/experiments/analyze_all.py`. |
| `data/cases/*.npz`, `data/structures.csv` | 22 protein pairs:<br>• contact graphs (`A_c8`, `B_c8`: Cα–Cα < 8 Å, sequence separation > 2);<br>• known correspondence `pi_true` (used for evaluation only);<br>• coordinates `x`, `y`;<br>• metadata. |
| `v3/code/gacert2/` | Two-phase estimator (`phase.py`), overlap windows (`windows.py`), baselines (`baselines2.py`, `umbrella.py`), variable selection (`bvs.py`), instances and witnesses (`instances.py`). |
| `v3/code/experiments/` | All experiment, analysis, table and figure scripts (see below). |
| `v3/results/` | Every per-instance result file behind the paper's numbers, plus `summary_*.json`. |
| `v3/figures/`, `v3/paper/tables/` | Generated figures and LaTeX tables. |

## Running experiments

Experiment scripts skip instances whose result file already exists. To recompute an experiment, delete its result directory first. The analysis, table and figure scripts alone regenerate every number from the included results.

## Single entry point (tables and statuses from saved results)

```
python v3/code/experiments/reproduce_tables.py
```

This command regenerates every table in `v3/paper/tables/` from the saved results in `v3/results/` without re-running any sampler (about 1 minute). It then prints:

- the released status of all 22 protein pairs under the fixed retry policy: one two-seed attempt, then at most one two-seed retry on the same output. The source is `results/summary_released_22.json`; the table is `tables/released.tex`.
- all 19 full-pipeline replicates, with first attempt, retry, final status and accuracy. The source is `results/summary_r5.json` → `runs`; the table is `tables/runs.tex`.

Every statistic is labelled with its type:

| Type | Statistics |
|---|---|
| Rigorous | the λ = 0 certificate; the witness floor |
| TI-based | the tilted bound; the two-phase occupancy error; the two-phase estimates |
| Heuristic screen | the validity-check status |

The same labels appear in the table rows ("rigorous cert.", "TI-based …") and in the captions.

## Reproducing the paper

Times are wall-clock for a single process on a 2-core x86-64 VM, usually with another job on the second core (never measured on an otherwise idle machine); they are indicative only.

| Paper item | Command | Output | Time |
|---|---|---|---|
| Unit tests | `python code/tests/test_core.py` | pass/fail | 15 s |
| Two-phase runs, reference instances | `python v3/code/experiments/run_twophase.py --case pscdb_O_19 --tag gt ...` (see `v3/code/experiments/queue_*.sh` for the exact argument lists used) | `v3/results/twophase/*_gt.json` | 1–3 min per run |
| Monte Carlo references ("gt" in file names) | `gt_mh.py`, `gt_umbrella.py`, `gt_windows.py` (in `v3/code/experiments/`) | `v3/results/gt/` | 10–60 min per instance |
| Baselines (appendix Table 13) | `python v3/code/experiments/baselines.py --case pscdb_N_141 --betas 1.5,1.51,1.52,1.53,1.55 --logB 26`<br>`python v3/code/experiments/baselines2.py --case pscdb_N_141 --betas 1.5,1.51,1.52,1.53,1.55 --logB 28` | `v3/results/baselines*/` | 10–20 min per instance |
| Appendix Table 13, appendix Figure 5 | `python v3/code/experiments/analyze_gt.py pscdb_N_74:mh pscdb_CL_64:umb pscdb_O_19:umb syn_n40_d6_q0.1_s1:umb pscdb_N_141:win4`<br>`python v3/code/experiments/fig_gt.py`<br>`python v3/code/experiments/make_tables.py` | `v3/results/gt_comparison.json`, `v3/figures/gt_phase.pdf`, `v3/paper/tables/gt.tex` | < 1 min |
| Variable selection (Table 2, Figure 2 left) | `python v3/code/experiments/exp_bvs.py`, then `make_tables.py`, then `fig_concept.py` | `v3/results/bvs/`, `tables/bvs.tex`, `figures/bvs_lambda.pdf` | 30 min |
| Silent regimes (Table 3, appendix Figure 6) | `python v3/code/experiments/exp_silent_bvs.py`<br>`python v3/code/experiments/exp_silent_graph.py 0 16`<br>`python v3/code/experiments/make_silent.py`<br>`python v3/code/experiments/fig_silent.py` | `v3/results/silent_bvs/`, `v3/results/silent_graph/`, `tables/silent.tex`, `results/summary_silent.json`, `figures/silent.pdf` | 5 min + 25 min |
| Validity check, 22 pairs (appendix table) | `validity.py`, `validity_c.py`, `window_budget.py`, then `make_tables.py` | `v3/results/validity*/`, `tables/valid.tex` | 2 h |
| Held-out validity test (appendix table) | `python v3/code/experiments/exp_heldout_validity.py 0 10`<br>`python v3/code/experiments/make_heldout_table.py` | `v3/results/heldout/`, `tables/heldout.tex` | 10 min |
| Proposition 1 check, Riemann bracket | `python v3/code/experiments/d1_check.py` | `v3/results/d1_check.json` | 1 min |
| Exact checks (appendix tables) | `exact_twophase.py`, `exact_nested.py` | `v3/results/exact_*.json` | 5 min |
| Partial phases, compact-patch control (Figure 4, appendix table) | `middle_phase.py`, `middle_control.py`, `rigid_control.py` | `v3/results/middle_phase/`, `v3/figures/adk_partial.pdf` | 10 min |
| Figure 1 | `python v3/code/experiments/fig_concept.py` | `v3/figures/profiles.pdf` | 5 min (cached profiles included) |
| Rugged synthetic regime (appendix Table 14) | `python v3/code/experiments/exp_alarm.py`, then `make_tables.py` | `v3/results/alarm/` | 10 min |
| Standard runs on proteins (Appendix D) | `rerun_v1_chains.py`, `analyze_v1chains.py` | `v3/results/v1chains/`, `summary_v1chains.json` | 1 h |
| End-to-end replicates, error budget, cost (appendix tables) | `python v3/code/experiments/pipeline_rep.py --case syn_n40_d6_q0.1_s1,pscdb_N_74,pscdb_CL_64,pscdb_O_19,pscdb_N_141 --seeds 1,2,3` | `v3/results/pipeline_reps/` | 25 min |
| Matched-total-cost comparators: broad and witness-seeded PT, extended WL, direct windows | `python v3/code/experiments/baselines3.py --case syn_n40_d6_q0.1_s1,pscdb_N_74,pscdb_CL_64,pscdb_O_19,pscdb_N_141 --logB 29` | `v3/results/baselines3/` | 45 min |
| Tables for the error budget, cost and comparators | `python v3/code/experiments/analyze_r5.py` | `v3/results/summary_r5.json`, `tables/{budget,cost,pt}.tex` | < 1 min |
| Round 6: extra pipeline replicates (5 runs for O.19 and N.141) | `python v3/code/experiments/pipeline_rep.py --case pscdb_O_19,pscdb_N_141 --seeds 4,5` | `v3/results/pipeline_reps/` | 6 min |
| Round 6: mesh halving | `python v3/code/experiments/mesh_half.py graphs`<br>`python v3/code/experiments/mesh_half.py bvs` | `v3/results/mesh_half/`, `tables/mesh.tex` | 13 min |
| Round 6: validity at five temperatures, B.44 window re-check | `python v3/code/experiments/validity_multi.py syn_n40_d6_q0.1_s1,pscdb_N_74,pscdb_CL_64,pscdb_O_19,pscdb_N_141,pscdb_B_44,pscdb_O_17,pscdb_N_139,pscdb_N_217`<br>`python v3/code/experiments/window_recheck.py pscdb_B_44` | `v3/results/validity_multi/`, `tables/validity_multi.tex` | 20 min |
| Round 6: replicated baselines (seeds 1, 2) | `python v3/code/experiments/baselines3.py --case syn_n40_d6_q0.1_s1,pscdb_N_74,pscdb_CL_64,pscdb_O_19,pscdb_N_141 --seeds 1,2 --methods pt_broad_w,wl_ext,win_direct` | `v3/results/baselines3/*_s{1,2}.json` | 50 min |
| Truncated-prior anchor check | `python v3/code/experiments/truncation_check.py` | `v3/results/truncation_anchor_check.json` | < 1 s |
| Figure 3 (main text) and appendix Figure 5 | `python v3/code/experiments/fig_gt.py --top` (phase-weight row, `gt_phase_top.pdf`) and `python v3/code/experiments/fig_gt.py` (both rows, `gt_phase.pdf`) | `v3/figures/gt_phase_top.pdf`, `v3/figures/gt_phase.pdf` | < 1 min |
| Figure and PDF metadata stripping | `python v3/code/experiments/strip_metadata.py v3/figures` | in place | < 1 min |

## Round 15: certificates for standard TI, structured floors, and the held-out evaluation (Appendices F and G)

Every number of Appendices F–G, main-text Table 4, Figure 2 and the protein certification in Section 5.1 (Q1) is regenerated from saved results, without re-running a sampler, by

```
python v3/code/experiments/r15_reproduce.py
```

which `reproduce_tables.py` also calls at the end. Every R15 result file carries `prov`: the script's SHA-256, the SHA-256 of every `gacert2/*.py`, start and end time and wall-clock seconds. Labels: RIGOROUS (deterministic floor), TI-BASED (two-phase or other Monte Carlo estimate, context only), EMPIRICAL (error against a Monte Carlo reference).

| Paper item | Experiment (re-runs samplers) | Analysis (saved results only) | Outputs |
|---|---|---|---|
| Standard TI certified by the witness floor; Tables 20, 21, 23; Figure 2 right | `r15_fpti_cert.py` (job list `r15_m1_jobs.txt`) | `r15_m1_analyze.py`, `r15_paper_tables.py` | `v3/results/r15/fpti_cert/`, `summary_fpti_cert.json`, `tables/r15_fpti_cert.tex`, `r15_fpti_silent.tex`, `r15_pairs.tex` |
| Extra seeds on the six returned pairs (Table 22) | `r15_fpti_cert.py` (job list `r15_x2_jobs.txt`) | `r15_m1_analyze.py`, `r15_paper_tables.py` | `summary_fpti_cert.json` → `X2_extra_seeds`, `tables/r15_x2.tex` |
| Coset (product-of-cosets) floor (Table 24); unit tests | `r15_mass_floor.py`; `r15_test_variational.py` | `r15_mass_floor_analyze.py` | `v3/results/r15/mass_floor/`, `summary_mass_floor.json`, `tables/r15_mass_floor.tex`; code in `v3/code/gacert2/variational.py` |
| Delay law, predictive check | – | `r15_delay_check.py` | `v3/results/r15/summary_delay.json` |
| PT-based TI (Table 25) | `r15_pt_ti.py` | `r15_pt_ti_analyze.py` | `v3/results/r15/pt_ti/`, `summary_pt_ti.json`, `tables/r15_pt_ti.tex` |
| Held-out evaluation: pre-registration | – | – | `v3/results/r15/p1_heldout/PREREG.json` (protocol, operational definitions, SHA-256 of 19 frozen files) |
| Held-out references, pipeline runs, comparators | `r15_p1_reference.py`, `r15_p1_pipeline.py`, `r15_p1_comparators.py` (queue `r15_p1_queue.py`) | `r15_p1_analyze.py` (frozen), `r15_p1_present.py`, `r15_p1_subsets.py`, `r15_paper_tables.py` | `v3/results/r15/p1_heldout/{ref,pipe,comp}/`, `summary_p1.json`, `summary_p1_subsets.json`; Table 4 (`r15_p1_main.tex`), Tables 26–28, 31; appendix Figure 7 |
| Comparators with a second seed (Table 29) | `r15_x1_comparators.py` | `r15_x1_analyze.py`, `r15_paper_tables.py` | `v3/results/r15/p1_heldout/comp_x1/`, `summary_x1.json`, `tables/r15_x1.tex` |
| Compute ledger (Table 30) | – | `r15_paper_tables.py` | `tables/r15_ledger.tex` |
| Figure 2 | – | `r15_fig_takeaway.py --coset` | `v3/figures/r15_takeaway.pdf` |
| Post-hoc diagnostics (not pre-registered) | – | – | `v3/results/r15/p1_heldout/posthoc_ref_check.json`, `posthoc_pt_init_check.json` |

Development smoke tests (development instances only) are in `v3/results/r15/smoke/`. Total round-15 compute: 37.2 CPU-hours on the 2-core machine (Table 30), plus about 0.5 CPU-hours of smoke tests and post-hoc checks.

## Released validity procedure (default)

`v3/code/gacert2/diagnose.py` implements the diagnostic as released. `pipeline_rep.py --rules transition` is the default.

- **Rules.** The five frozen rules are evaluated only at the estimated transition β̂_c:
  - edge occupancy < 1%;
  - D two-start;
  - C two-start;
  - window profile |Δlogit| < 1;
  - middle basin.
- **Two seeds.** The whole check runs with two independent seeds.
  - The status is `pass` only if every rule passes in both seeds.
  - It is `multi-phase` only if the middle-basin rule fires in both seeds (a basin of mass above 5% with peak in (0.1n, k0 + 0.05n)).
  - Otherwise it is `unresolved`: a rule failed, or the two seeds disagree.
- **Bounded retry (released operational policy, `diagnose_bounded`).**
  - After an `unresolved` first attempt, the two-seed check is repeated **once** with two fresh seeds on the same estimator output (same β̂_c, witness and estimate). The estimator is not re-run, and no budget is added.
  - A pass on the retry is reported as `pass (retry)`. Otherwise the output stays `unresolved` (or `multi-phase` if the middle-basin rule fires in both retry seeds).
  - There is no further retry and no "repeat until pass" mode. A retry can accept a favourable fluctuation, which is why it is bounded and recorded.
- **A pass is a screen, not a validation.** A passing output is a screened estimate. The checks use dedicated restricted chains started at β̂_c, not the saved TI trajectories, so they do not show that the integrated trajectories mixed.
- **No two-phase β_c is reported unless the status is `pass`.**
- **A failed rule is a diagnostic failure, not proof of an extra phase.**
- **Extended mode.** `--rules extended` and `validity_multi.py` add the optional off-transition sensitivity analysis (β̂_c ± 0.1, ± 0.25). It uses two extensions that are not part of the frozen rules:
  - region rules count only for regions with ≥ 1% of the estimated mass;
  - the window rule also passes when both weights are < 0.01 or both > 0.99.
- **Survey vs release.**
  - The 22-pair survey used one seed, plus a second window pass for inconclusive pairs: 8 pass / 5 multi-phase / 9 unresolved.
  - The released two-seed procedure gives 5 / 5 / 12.
  - The exhaustive released-status table (the appendix table "Released procedure on all 22 pairs") is `tables/released.tex`, with its summary in `results/summary_released_22.json`.

## Round-7 audit commands

| Item | Command | Output |
|---|---|---|
| Exact capped-prior anchors: brute-force check and tail sizes 2.35e-53 / 7.64e-32 | `python v3/code/experiments/test_capped_anchor.py` | printed; asserts |
| Per-replicate diagnostic status (default rules, two seeds) for every full-pipeline replicate | `python v3/code/experiments/replicate_status.py` | `v3/results/replicate_status/*.json` |
| Per-run status table, per-run costs, error budget, mesh halving, comparators | `python v3/code/experiments/analyze_r5.py` | `v3/results/summary_r5.json`; `tables/{runs,budget,cost,pt,mesh,validity_multi}.tex` |
| Mesh-halving raw results | `python v3/code/experiments/mesh_half.py graphs` and `... bvs` | `v3/results/mesh_half/` |
| Round 10: bounded retry for the unresolved full-pipeline replicates | `python v3/code/experiments/retry_status.py` | `v3/results/retry_status/*.json` |
| Round 10: released two-seed check on the five survey multi-phase pairs | `python v3/code/experiments/middle_recheck.py` | `v3/results/middle_recheck/*.json` |
| Round 10: bounded retry on the 12 pairs unresolved after the first attempt | `python v3/code/experiments/survey_retry.py` | `v3/results/survey_retry/*.json` |
| Round 10: exhaustive released-status table | `python v3/code/experiments/make_tables.py` | `tables/released.tex`, `results/summary_released_22.json` |

### Per-run results behind appendix Tables 15–19

All per-run results are kept, including failed checks, refinement restarts, every accuracy outcome, costs and times. No run is dropped.

| Table | Content | Per-run files |
|---|---|---|
| 15 | Mesh halving | `v3/results/mesh_half/*.json`: every run, both meshes |
| 16 | End-to-end replicates and error budget | `v3/results/pipeline_reps/*.json`: per replicate — witness score, restarts, β̂_c, phase weights on the grid, SEs, cost, seconds |
| 17 | Per-run status | `v3/results/pipeline_reps/*.json` together with:<br>• `v3/results/replicate_status/*.json` (first two-seed attempt: rules and statistics per seed, proposals, seconds)<br>• `v3/results/retry_status/*.json` (retry)<br>• `v3/results/summary_r5.json` → `runs` |
| 18 | Proposals per run with one diagnostic seed | `pipeline_reps/*.json` → `cost`; `summary_r5.json` → `cost` |
| 19 | Comparators and wall-clock | `v3/results/baselines3/*.json` (every comparator run and seed, with seconds); `v3/results/validity_multi/*.json` (one-seed check times); `summary_r5.json` → `comparators` |

All tables are regenerated by `python v3/code/experiments/make_tables.py` followed by `python v3/code/experiments/analyze_r5.py`.

## Seeds

Every script fixes its seeds per instance: from the instance parameters for synthetic graphs and designs, and from explicit `--seed` arguments or fixed constants for proteins. Results are deterministic given the library versions in `requirements.txt`, up to floating-point summation order in Numba.

## Data licences

Protein coordinates are derived from the Protein Data Bank; PDB data are CC0 1.0. The structure pairs come from PSCDB (Amemiya et al., Nucleic Acids Research 40(D1):D554–D558, 2012), licensed under CC BY-SA 4.0. The PSCDB licence requires this credit line:

> PSCDB © Amemiya Takayuki (The Molecular Profiling Research Center for Drug Discovery (molprof), The National Institute of Advanced Industrial Science and Technology (AIST)) licensed under CC Attribution-Share Alike 4.0 International

The PSCDB-derived contact graphs and metadata in `data/` are released under CC BY-SA 4.0, with this attribution (also in `data/NOTICE`) and attribution to the PDB depositors of each entry listed in `data/structures.csv`.

## Licence

The code is released under the MIT License in the repository root (`../LICENSE`). The PSCDB-derived data in `data/` are released under CC BY-SA 4.0, as stated above and in `data/NOTICE`.
