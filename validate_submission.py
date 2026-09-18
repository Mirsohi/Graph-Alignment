from __future__ import annotations
from pathlib import Path
import math, re, sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
errors: list[str] = []

def check(condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)

# Exact-oracle budget grid
b = pd.read_csv(ROOT / "results/budget_curves/budget_curve_raw.csv")
check(len(b) == 600, f"budget rows {len(b)} != 600")
counts = b.groupby(["n", "method", "local_proposals_per_chain"]).size()
check((counts == 20).all(), "not every size/method/budget cell has 20 instances")

f = pd.read_csv(ROOT / "results/floor_initialization/iid_floor_raw.csv")
check(len(f) == 600, f"floor rows {len(f)} != 600")
check((f.iid_replicates == 500).all(), "i.i.d. floor replicates are not all 500")

m = b[b.local_proposals_per_chain == 262144].groupby(["n", "method"]).mean_row_tv.mean()
check(abs(m[(10, "local")] - 0.1291570277) < 5e-9, "n=10 local headline mismatch")
check(abs(m[(10, "parallel_tempering")] - 0.00462885725) < 5e-9, "n=10 PT headline mismatch")
ff = f[f.local_proposals_per_chain == 262144].groupby(["n", "method"]).iid_floor_mean.mean()
check(abs(ff[(10, "parallel_tempering")] - 0.000696746) < 5e-7, "n=10 PT floor mismatch")

smc = pd.read_csv(ROOT / "results/inference_baselines/inference_baselines_summary.csv")
smc10 = float(smc.loc[smc.n == 10, "mean_row_tv"].iloc[0])
check(abs(smc10 - 0.0136993769) < 5e-9, "n=10 SMC headline mismatch")

d = pd.read_csv(ROOT / "results/diagnostic_summary.csv").set_index("diagnostic")
check(d.loc["marginal disagreement", "spearman_rho"] > d.loc["score split-Rhat", "spearman_rho"], "diagnostic ranking mismatch")

c = pd.read_csv(ROOT / "results/family_replication/family_chain_results.csv")
large = c[c.n >= 20]
check(int(large[large.method == "local"].traverses_both_orientations.sum()) == 0, "large local traversal count mismatch")
check(int(large[large.method == "near_twin_jump"].traverses_both_orientations.sum()) == 240, "large jump traversal count mismatch")

bound = math.exp(2 * (30 - 1)) / (4 * math.factorial(14) ** 2 * math.comb(14, 7) ** 2)
check(abs(bound - 4.316270556e-5) < 1e-10, f"finite bound mismatch: {bound}")

# Basic anonymity scan over text/code files.
for folder in [ROOT / "code", ROOT / "results/symmetry_stress_test"]:
    for path in folder.rglob("*"):
        if path.name == "validate_submission.py":
            continue
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".txt", ".sh", ""}:
            text = path.read_text(errors="ignore")
            check(not re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text), f"email address in {path.relative_to(ROOT)}")

if errors:
    print("VALIDATION FAILED")
    for error in errors:
        print("-", error)
    sys.exit(1)

print("VALIDATION PASSED")
print("- 30 budget cells, each with 20 graph pairs")
print("- n=10 local/PT/floor:", m[(10, "local")], m[(10, "parallel_tempering")], ff[(10, "parallel_tempering")])
print("- n=10 population SMC:", smc10)
print("- near-twin traversal for n>=20: 0/240 local, 240/240 jump")
