#!/bin/sh
cd "$(dirname "$0")/../.."
while pgrep -f "gt_umbrella.py" > /dev/null; do sleep 20; done
python code/experiments/gt_umbrella.py --case pscdb_N_74 --betas 1.40,1.44,1.47,1.50,1.52,1.54,1.57,1.60,1.65 --logT 27
python code/experiments/synth_pipeline.py syn_n40_d6_q0.1_s0 syn_n40_d6_q0.1_s1 syn_n60_d6_q0.1_s0
