#!/bin/sh
cd "$(dirname "$0")/../.."
BETAS=1.49,1.5,1.51,1.52,1.53,1.55
python code/experiments/run_twophase.py --case pscdb_N_141 --reps 2 --logT 18 --tag gt --marg_at $BETAS
python code/experiments/fix_reference.py gt pscdb_N_141
python code/experiments/gt_wl.py pscdb_N_141 $BETAS 30 2
python code/experiments/gt_umbrella.py --case pscdb_N_141 --betas $BETAS --logT 28 --seeds 2
python code/experiments/baselines.py --case pscdb_N_141 --betas $BETAS --logB 26
python code/experiments/baselines2.py --case pscdb_N_141 --betas $BETAS --logB 28
