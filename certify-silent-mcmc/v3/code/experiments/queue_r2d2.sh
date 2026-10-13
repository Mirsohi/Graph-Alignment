#!/bin/sh
cd "$(dirname "$0")/../.."
while ps aux | grep -v grep | grep -q "run_twophase.py --case pscdb_N_141"; do sleep 15; done
BETAS=1.49,1.5,1.51,1.52,1.53,1.55
python code/experiments/fix_reference.py gt pscdb_N_141
python code/experiments/gt_wl.py pscdb_N_141 $BETAS 29 2
python code/experiments/gt_umbrella.py --case pscdb_N_141 --betas $BETAS --logT 27 --seeds 2
python code/experiments/baselines.py --case pscdb_N_141 --betas $BETAS --logB 25
python code/experiments/baselines2.py --case pscdb_N_141 --betas $BETAS --logB 27
