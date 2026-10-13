#!/bin/sh
cd "$(dirname "$0")/../.."
python code/experiments/validity.py all adk_crystal_control,pscdb_CD_21,pscdb_N_139,pscdb_N_261,pscdb_N_271,pscdb_N_288,pscdb_N_75,pscdb_N_82 --T 22 --seed 2 --out validity22
