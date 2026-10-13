#!/bin/sh
cd "$(dirname "$0")/../.."
while pgrep -f "rerun_v1_chains" > /dev/null; do sleep 15; done
python code/experiments/window_budget.py adk_crystal_control 1.746 64 21,22,23
python code/experiments/window_budget.py pscdb_CD_21 1.721 70 22,23
