#!/bin/sh
cd "$(dirname "$0")/../.."
python code/experiments/baselines.py --case pscdb_N_141 --betas 1.5,1.51,1.52,1.53,1.55 --logB 26
python code/experiments/baselines2.py --case pscdb_N_141 --betas 1.5,1.51,1.52,1.53,1.55 --logB 28
