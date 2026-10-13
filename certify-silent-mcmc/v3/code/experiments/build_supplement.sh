#!/bin/sh
# Build the anonymous supplement archive v3/supplement/code_and_data.zip (round 15).
# Layout: code/ (core package, tests, earlier scripts), data/, v3/{code,results,figures,paper/tables}.
# Excluded: results/r15/**/locks (empty queue lock directories) and the stale backup twophase_singleref_backup.
set -e
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
V=v3
STAGE=${TMPDIR:-/tmp}/supp_stage_$$
rm -rf "$STAGE"; mkdir -p "$STAGE/code_and_data/$V/paper"
S="$STAGE/code_and_data"
cp -r "$ROOT/code" "$S/code"
cp -r "$ROOT/data" "$S/data"
cp -r "$ROOT/$V/code" "$S/$V/code"
cp -r "$ROOT/$V/figures" "$S/$V/figures"
cp -r "$ROOT/$V/paper/tables" "$S/$V/paper/tables"
mkdir -p "$S/$V/results"
( cd "$ROOT/$V/results" && tar cf - --exclude=twophase_singleref_backup --exclude=locks . ) | ( cd "$S/$V/results" && tar xf - )
find "$S" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$S" -name '*.pyc' -delete
# queue scripts carry absolute build paths; make them relative
for f in "$S"/$V/code/experiments/*.sh "$S"/$V/code/experiments/*.sh.part2; do
  [ -f "$f" ] && sed -i -E "s#cd $ROOT/v[23]#cd \"\$(dirname \"\$0\")/../..\"#; s#cd /mnt/[A-Za-z_./-]*/AISTATS27/v[23]#cd \"\$(dirname \"\$0\")/../..\"#; s#$ROOT#.#g; s#/mnt/[A-Za-z_./-]*/AISTATS27#.#g" "$f"
done
python "$ROOT/$V/code/experiments/strip_metadata.py" "$S"
cp "$ROOT/$V/supplement/README.md" "$ROOT/$V/supplement/requirements.txt" "$S/"
( cd "$STAGE" && rm -f "$ROOT/$V/supplement/code_and_data.zip" && zip -qrX "$ROOT/$V/supplement/code_and_data.zip" code_and_data )
echo "staged at $STAGE"
