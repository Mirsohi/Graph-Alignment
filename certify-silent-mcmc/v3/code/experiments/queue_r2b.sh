#!/bin/sh
cd "$(dirname "$0")/../.."
# adaptive k0 at the first dividing minimum of the overlap profile (diffuse | aligned)
python code/experiments/run_twophase.py --case adk_open_closed --reps 2 --logT 17 --k0frac 0.0467 --tag adapt
python code/experiments/run_twophase.py --case pscdb_IL_55 --reps 2 --logT 17 --k0frac 0.0462 --tag adapt
python code/experiments/run_twophase.py --case pscdb_IL_74 --reps 2 --logT 17 --k0frac 0.0423 --tag adapt
python code/experiments/run_twophase.py --case pscdb_O_16 --reps 2 --logT 17 --k0frac 0.0779 --tag adapt
python code/experiments/run_twophase.py --case pscdb_N_185 --reps 2 --logT 17 --k0frac 0.1 --tag adapt
python code/experiments/fix_reference.py adapt all
