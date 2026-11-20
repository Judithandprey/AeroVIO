#!/usr/bin/env bash
# No-hover training flights for the MLP retrain (stage 2 scale experiment).
# Scenes and seeds are disjoint from every evaluation and earlier training set:
#   public ProcTHOR-Train-1, old train 2/3/4, old validation 5/6,
#   val-agile 40/60/80/100/120/140 (seed 2000), val-nohover 50/90/130/170 (seed 3000).
# Resumable: a scene whose zips already exist is skipped.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=data/train-nohover
TRAIN_SCENES=(7 13 19 25 31 37 43 49 55 61 67 73 79 85 91 97 103 109 115 121 127 133 139 145 151 157 163 169 175 181 187 193)
VAL_SCENES=(10 70 110 150)
run() {  # split seed_base index scene
  local split=$1 seed=$2 scene=$3 dir=$OUT/$1
  mkdir -p "$dir"
  if [ -f "$dir/sim-ProcTHOR-Train-$scene-0001.zip" ]; then echo "skip $scene"; return; fi
  echo "=== $(date -Is) $split scene $scene seed $seed"
  $PY -m viobench simulate -o "$dir" --scene-names "ProcTHOR-Train-$scene" --episodes 2 \
      --seed "$seed" --hover 0 --no-video || echo "FAILED scene $scene"
}
i=0; for s in "${VAL_SCENES[@]}"; do run validation $((8000 + 10 * i)) "$s"; i=$((i + 1)); done
i=0; for s in "${TRAIN_SCENES[@]}"; do run train $((7000 + 10 * i)) "$s"; i=$((i + 1)); done
echo "GEN_DONE $(date -Is)"
