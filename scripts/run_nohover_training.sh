#!/usr/bin/env bash
# Overnight pipeline for the no-hover MLP retrain (experiments/nohover-v11).
# Waits for both data generators, builds caches, checks that the regenerated
# original data reproduces the shipped checkpoint, then trains two variants.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.train-venv/bin/python
EXP=experiments/nohover-v11
LOG=logs/nohover
mkdir -p "$EXP"

echo "$(date -Is) waiting for data generation"
until grep -q GEN_DONE $LOG/gen-nohover.log 2>/dev/null; do sleep 120; done
until grep -q '"status": "complete"' data/hyperdrone-motion-v2/data_manifest.json 2>/dev/null; do
  if ! pgrep -f generate_training_data.py > /dev/null; then echo "old-data generator died"; exit 1; fi
  sleep 120
done

echo "$(date -Is) preparing original cache"
[ -d data/motion-cache-v2 ] || $PY scripts/prepare_motion_data.py \
  --manifest data/hyperdrone-motion-v2/data_manifest.json --out data/motion-cache-v2

echo "$(date -Is) reproduction check: original recipe on regenerated data"
[ -d $EXP/repro-compact ] || $PY scripts/train_motion.py --cache data/motion-cache-v2 --out $EXP/repro-full \
  --compact-out $EXP/repro-compact --input-mode imu_only --warp-weight 2 --epochs 40 --seed 17 --compact-tolerance 1e-4
python3 - <<'EOF'
import json
new = json.load(open("experiments/nohover-v11/repro-compact/meta.json"))["model_sha256"]
old = json.load(open("experiments/motion-ablation-v2/imu-only-compact/meta.json"))["model_sha256"]
print("REPRO", "IDENTICAL" if new == old else "DIFFERENT", new, old)
EOF

echo "$(date -Is) preparing merged cache"
[ -d data/motion-cache-nohover-v11 ] || $PY scripts/prepare_nohover_cache.py --old-cache data/motion-cache-v2 \
  --nohover data/train-nohover --out data/motion-cache-nohover-v11 \
  --balanced-plan data/motion-cache-nohover-v11-balanced-plan.npz --old-fraction 0.25

echo "$(date -Is) training uniform variant"
[ -d $EXP/uniform-compact ] || $PY scripts/train_motion.py --cache data/motion-cache-nohover-v11 \
  --out $EXP/uniform-full --compact-out $EXP/uniform-compact --input-mode imu_only --warp-weight 2 --epochs 40 --seed 17 --compact-tolerance 1e-4
echo "$(date -Is) training balanced variant"
[ -d $EXP/balanced-compact ] || $PY scripts/train_motion.py --cache data/motion-cache-nohover-v11 \
  --out $EXP/balanced-full --compact-out $EXP/balanced-compact --input-mode imu_only --warp-weight 2 --epochs 40 --seed 17 --compact-tolerance 1e-4 \
  --sampling-plan data/motion-cache-nohover-v11-balanced-plan.npz
echo "TRAIN_DONE $(date -Is)"
