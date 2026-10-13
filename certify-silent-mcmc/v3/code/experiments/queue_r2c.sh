#!/bin/sh
cd "$(dirname "$0")/../.."
python code/experiments/baselines2.py --case pscdb_O_19 --betas 1.4,1.44,1.45,1.46,1.47,1.48,1.52 --logB 28
python code/experiments/baselines2.py --case pscdb_CL_64 --betas 1.42,1.46,1.47,1.48,1.49,1.5,1.54 --logB 28
python code/experiments/baselines2.py --case syn_n40_d6_q0.1_s1 --betas 1.797,1.847,1.862,1.877,1.892,1.907,1.957 --logB 28
python code/experiments/baselines2.py --case pscdb_N_74 --betas 1.4,1.44,1.47,1.5,1.52,1.54,1.57,1.6,1.65 --logB 28
python -W ignore code/experiments/exp_alarm.py
