#!/bin/sh
cd "$(dirname "$0")/../.."
while ps aux | grep -v grep | grep -q "queue_r2d2"; do sleep 20; done
python code/experiments/gt_wl.py pscdb_N_141 1.49,1.5,1.51,1.52,1.53,1.55 29 2
while ps aux | grep -v grep | grep -q "queue_r2c"; do sleep 20; done
python code/experiments/baselines2.py --case pscdb_O_19 --betas 1.4,1.44,1.45,1.46,1.47,1.48,1.52 --logB 28
