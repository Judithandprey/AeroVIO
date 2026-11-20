#!/usr/bin/env bash
# Evaluate the anti-runaway variants (V8 + tilt K=0.2 + bias g=0.2, old weights) on four sets.
# The old-weights baseline on val-nohover/val-agile/public is results/leo/bias/*-bias-g0.2.
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -q GEN_DONE logs/runaway/gen-fast.log; do sleep 60; done
OUT=results/leo/runaway
mkdir -p $OUT
run() {  # name wasm set
  local eps; case $3 in nohover) eps=data/val-nohover;; agile) eps=data/val-agile;; fast) eps=data/val-fast;; pub) eps="";; esac
  [ -f $OUT/$3-$1/report.json ] || .venv/bin/python -m viobench evaluate $2 $eps --limits configs/evaluation-limits.json \
    -o $OUT/$3-$1 > $OUT/$3-$1.log 2>&1
  echo "$(date -Is) $3-$1: $(tail -n 1 $OUT/$3-$1.log)"
}
run base artifacts/aerovio-v8-tilt-bias-g0.2.wasm fast
for v in speedcap2 speedcap3 visdamp scale0.7; do
  w=artifacts/aerovio-v8-tilt-bias-g0.2-$v.wasm
  .venv/bin/python -m viobench check $w --frames 40 | tail -n 1
  for s in fast nohover agile pub; do run $v $w $s; done
done
echo EVAL_DONE
