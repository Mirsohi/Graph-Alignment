#!/bin/sh
# after N.74 MH ground truth: estimator + baselines on N.74, then on CL.64 / O.19
cd "$(dirname "$0")/../.."
while pgrep -f "gt_mh.py --case pscdb_N_74" > /dev/null; do sleep 20; done
python code/experiments/run_twophase.py --case pscdb_N_74 --reps 4 --tag gt --marg_at 1.40,1.44,1.47,1.50,1.52,1.54,1.57,1.60,1.65
python code/experiments/baselines.py --case pscdb_N_74 --betas 1.40,1.44,1.47,1.50,1.52,1.54,1.57,1.60,1.65 --logB 27
python code/experiments/run_twophase.py --case pscdb_CL_64 --reps 4 --tag gt --marg_at 1.42,1.46,1.47,1.48,1.49,1.50,1.54
python code/experiments/run_twophase.py --case pscdb_O_19 --reps 4 --tag gt --marg_at 1.40,1.44,1.45,1.46,1.47,1.48,1.52
python code/experiments/baselines.py --case pscdb_CL_64 --betas 1.42,1.46,1.47,1.48,1.49,1.50,1.54 --logB 27
python code/experiments/baselines.py --case pscdb_O_19 --betas 1.40,1.44,1.45,1.46,1.47,1.48,1.52 --logB 27
python code/experiments/switch_counts.py --case pscdb_CL_64 --betas 1.42,1.46,1.47,1.48,1.49,1.5,1.54 --logT 28
python code/experiments/switch_counts.py --case pscdb_O_19 --betas 1.4,1.44,1.45,1.46,1.47,1.48,1.52 --logT 28
