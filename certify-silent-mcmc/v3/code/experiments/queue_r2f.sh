#!/bin/sh
cd "$(dirname "$0")/../.."
python -W ignore code/experiments/exp_bvs.py decoy20 decoy20_s1 decoy20_s2 decoy20_s3 decoy20_s4 decoy20_s5 decoy200 decoy200_s1 decoy200_s2 decoy200_s3 decoy200_s4 decoy200_s5
# k0 / beta_hi sensitivity with independent seeds
python -c "
import sys; sys.path.insert(0,'code/experiments'); import run_twophase as r
for c in ['pscdb_O_19','pscdb_N_74','pscdb_CL_64']:
    r.run(c, 2, 17, 0.2, [], 'c8', 6.0, 'k02i', seed0=21)
    r.run(c, 2, 17, 0.5, [], 'c8', 6.0, 'k05i', seed0=22)
    r.run(c, 2, 17, 0.3, [], 'c8', 8.0, 'bhi8i', seed0=23)
    r.run(c, 2, 17, 0.3, [], 'c8', 6.0, 'basei', seed0=24)
"
for t in k02i k05i bhi8i basei; do python code/experiments/fix_reference.py $t all; done
