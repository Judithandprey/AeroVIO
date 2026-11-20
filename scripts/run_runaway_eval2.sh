#!/usr/bin/env bash
# val-narrow (fast flights, FOV 85 +/- 10 deg) and the fixed-camera variant.
set -uo pipefail
cd "$(dirname "$0")/.."
i=0; for s in 15 35 75 125 155 185; do
  .venv/bin/python scripts/simulate_fast.py --cruise 2.5 4.0 --accel 4.0 --jerk 12 --yaw-rate 2.5 -- -o data/val-narrow \
    --scene-names ProcTHOR-Train-$s --episodes 2 --seed $((4500+10*i)) --hover 0 --no-video --fov 85 --fov-range 10 || echo "FAILED $s"
  i=$((i+1)); done
echo GEN_DONE
OUT=results/leo/runaway
run() {  # name set
  local eps w=artifacts/aerovio-v8-tilt-bias-g0.2${1:+-$1}.wasm n=${1:-base}
  case $2 in nohover) eps=data/val-nohover;; agile) eps=data/val-agile;; fast) eps=data/val-fast;; narrow) eps=data/val-narrow;; pub) eps="";; esac
  [ -f $OUT/$2-$n/report.json ] || .venv/bin/python -m viobench evaluate $w $eps --limits configs/evaluation-limits.json -o $OUT/$2-$n > $OUT/$2-$n.log 2>&1
  echo "$(date -Is) $2-$n: $(tail -n 1 $OUT/$2-$n.log)"
}
for v in "" fixedcam visdamp scale0.7 speedcap2; do run "$v" narrow; done
for s in fast nohover agile pub; do run fixedcam $s; done
echo EVAL_DONE
