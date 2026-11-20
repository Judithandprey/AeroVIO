#!/usr/bin/env bash
# Build V8 + tilt K=0.2 + online bias g=0.2 with each retrained MLP and evaluate
# it on val-nohover, val-agile and public. Old-weights baseline reports are
# results/leo/bias/{nohover,agile,pub}-bias-g0.2 (same artifact hash cdc07749).
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs/nohover
until grep -q TRAIN_DONE $LOG/train.log 2>/dev/null; do
  if grep -qE "Error|Traceback|died" $LOG/train.log 2>/dev/null; then echo "training failed"; exit 1; fi
  sleep 120
done
TC="$PWD/workspace/build/toolchain/wasi-sdk/share/cmake/wasi-sdk-p1.cmake"
OUT=results/leo/nohover-weights
mkdir -p $OUT artifacts
for variant in uniform balanced; do
  d=workspace/visual_native/build/nohover-$variant
  cmake -S workspace/visual_native -B $d -DCMAKE_TOOLCHAIN_FILE="$TC" \
    -DVIOBENCH_NATIVE_DIR="$PWD/workspace/runtime/native" -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
    -DVISUAL_ROTATION_CORRECTION=ON -DLEO_GRAVITY_TILT=ON -DLEO_GRAVITY_TILT_GAIN=0.2 \
    -DLEO_ONLINE_BIAS=ON -DLEO_ONLINE_BIAS_GAIN=0.2 \
    -DLEO_WEIGHTS_NOHOVER=ON -DLEO_WEIGHTS_NOHOVER_CHECKPOINT="$PWD/experiments/nohover-v11/$variant-compact"
  cmake --build $d
  wasm=artifacts/aerovio-v8-tilt-bias-g0.2-nohover-$variant.wasm
  cp $d/solution.wasm $wasm
  sha256sum $wasm
  .venv/bin/python -m viobench check $wasm --frames 40
  for set in nohover agile pub; do
    case $set in nohover) eps=data/val-nohover;; agile) eps=data/val-agile;; pub) eps="";; esac
    [ -f $OUT/$set-$variant/report.json ] || .venv/bin/python -m viobench evaluate $wasm $eps \
      --limits configs/evaluation-limits.json -o $OUT/$set-$variant > $OUT/$set-$variant.log 2>&1
    tail -n 1 $OUT/$set-$variant.log
  done
done
.venv/bin/python scripts/summarize_eval.py \
  results/leo/bias/nohover-bias-g0.2 $OUT/nohover-uniform $OUT/nohover-balanced \
  results/leo/bias/agile-bias-g0.2 $OUT/agile-uniform $OUT/agile-balanced \
  results/leo/bias/pub-bias-g0.2 $OUT/pub-uniform $OUT/pub-balanced --json $OUT/summary.json | tee $OUT/summary.md
echo "EVAL_DONE $(date -Is)"
