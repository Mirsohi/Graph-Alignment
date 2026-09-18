# Graph-Alignment

Code and saved results for **Auditing Bayesian Graph Alignment: Diagnostic Comparisons and Reference Failure**, Melika Gorgi and Kourosh Mirsohi.

Intended repository: https://github.com/Mirsohi/Graph-Alignment

## Setup and quick checks

Use Python 3.13 (experiments used 3.13.9). From this repository root:

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
python extension/test_audit.py
python extension/test_followups.py
python extension/test_review_kernel.py
python extension/validate_results.py
python extension/validate_followups.py
```

## Results and analysis

`extension/results/exact/` and `large/` contain the 240-case exact and 240-case large tiers. `followup/` contains reference-budget and diagnostic studies. `review_revision/` contains 720 split-window records and 64 broader-kernel chains (16 configurations). `results/` contains the separate implementation-check inputs and supporting experiments. Saved outputs are included so inspection does not require resampling.

```sh
python extension/analyze.py
python extension/analyze_followups.py
python extension/followup_witness.py
python extension/analyze_review_revision.py
```

The release analysis omits the original machine-specific manuscript-builder call; numerical analysis is unchanged. `paper.tex` is the standalone current manuscript; compile with pdflatex three times. Analysis outputs do not overwrite that frozen manuscript. The figures it needs are in `figures/`.

## Sampling

The entry points are `extension/run_experiments.py`, `extension/run_followups.py`, and `extension/run_review_revision.py`. Inspect their `--help` for options. Full experiments can require substantial local compute. Completed JSON markers are resumable; keep paired NPZ files with them. Sampling-source hashes prevent mixing changed protocols. To run a different protocol, use a fresh result directory/copy. Never overwrite saved results merely to inspect the paper.

Exact errors and later drift are different endpoints. Follow-up graphs are failure-selected, intervals are pointwise, and the biological discussion is prospective. See the paper for full definitions and qualifications.

## Upload

Upload the contents of this directory to the repository root. No environment, unpublished source paper, review correspondence, or credentials are included. A software license has not been selected; the authors can add their chosen license before public release.

`MANIFEST.json` records SHA-256 hashes for all release files except itself. Historical metadata home-directory names have been normalized. Numerical arrays and sampling code are preserved.
