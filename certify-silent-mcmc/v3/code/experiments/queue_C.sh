#!/bin/sh
cd "$(dirname "$0")/../.."
while pgrep -f "queue_A.sh" > /dev/null; do sleep 30; done
CASES=$(ls ../data/cases/*.npz | xargs -n1 basename | sed 's/.npz//' | tr '\n' ',' | sed 's/,$//')
python code/experiments/run_twophase.py --case $CASES --reps 2 --logT 17 --tag all
python code/experiments/run_twophase.py --case pscdb_N_74,pscdb_O_19 --reps 2 --logT 17 --k0frac 0.2 --tag k02
python code/experiments/run_twophase.py --case pscdb_N_74,pscdb_O_19 --reps 2 --logT 17 --k0frac 0.5 --tag k05
python code/experiments/run_twophase.py --case pscdb_N_74,pscdb_O_19 --reps 2 --logT 17 --bhi 8 --tag bhi8
python -W ignore code/experiments/exp_bvs.py exact20_strong_s1 exact20_strong_s2 exact20_strong_s3 exact20_strong_s4 exact20_strong_s5 p200_s2 p200_s3 p200_s4 p200_s5 p200_s6 exact20_weak_s1 exact20_weak_s2
