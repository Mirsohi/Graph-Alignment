#!/bin/sh
# synthetic: umbrella ground truth (MH GT too noisy at n=40-60), then estimator + baselines
cd "$(dirname "$0")/../.."
while pgrep -f "synth_pipeline.py" > /dev/null; do sleep 20; done
python code/experiments/gt_umbrella.py --case syn_n40_d6_q0.1_s1 --betas 1.797,1.847,1.862,1.877,1.892,1.907,1.957 --logT 27
python code/experiments/run_twophase.py --case syn_n60_d6_q0.1_s0,syn_n60_d6_q0.1_s1 --reps 1 --tag pilot
