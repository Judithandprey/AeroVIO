# No-hover MLP retrain (v11, 2026-11-20)

The same 14→64→6 IMU-only network as `motion-ablation-v2/imu-only-compact`, with the same
recipe, retrained on the original data plus pose-only no-hover flights. The build uses it with
`-DLEO_WEIGHTS_NOHOVER=ON`, which reads `compact/` here and generates `weights_nohover.h`.

## Why

The network sees one 50 ms IMU window (preintegrated from zero velocity) and never its own
velocity. So its translation is close to the average speed of its training data. The original
data is 12 × 10 s flights that start with a 1 s hover (median speed 0.10 m/s). On no-hover
flights (true 0.28 m/s) it predicted 0.15 m/s.

## Data

- **Original:** `data/hyperdrone-motion-v2`, regenerated on this machine with
  `scripts/generate_training_data.py --config configs/training-data-v1.json`.
  - Every `episode.json` (trajectory + IMU) hash matches the original run. The rendered
    depth/RGB differ slightly (the renderer changed).
  - Reproduction check (`repro-*`): the original recipe on this data gives best epoch 14,
    validation pose loss 0.2585 (original 0.2583) and translation RMSE 1.064 cm (original 1.065).
    The ONNX hash is not identical.
- **No-hover:** `scripts/generate_nohover_data.sh` runs
  `viobench simulate --hover 0 --no-video`, 2 × 60 s flights per scene.
  - Train: ProcTHOR-Train-7, 13, 19, …, 193 (every 6th), seed 7000 + 10·i. Scene 121 crashed
    in all 6 attempts and was skipped, so 31 scenes, 62 flights, 74,338 pairs.
  - Validation (checkpoint selection only): scenes 10/70/110/150, seed 8000 + 10·i, 9,592 pairs.
  - Disjoint from public (1), the original split (2–6), val-agile (40–140, seed 2000) and
    val-nohover (50/90/130/170, seed 3000). `prepare_nohover_cache.py` refuses any overlap.
- **Merged cache:** `scripts/prepare_nohover_cache.py`.
  - The no-hover pairs have no depth, so their images, depth and visibility are zero.
    `photometric_loss` skips them and they get the pose loss only. The original pairs keep
    pose + 2× warp.

## Training

`scripts/run_nohover_training.sh`: `train_motion.py --input-mode imu_only --warp-weight 2 --epochs 40
--seed 17`, on CPU (about 4 min). Validation is the original 796 pairs plus the 9,592 no-hover
validation pairs.

| Variant | Sampling | Best epoch | Val pose loss | Val translation RMSE |
| --- | --- | ---: | ---: | ---: |
| `uniform` (**chosen**, copied to `compact/`) | all 76,726 pairs, uniform | 33 | 0.2763 | 1.072 cm |
| `balanced` | 25% of each epoch from the original pairs | 39 | 0.2772 | 1.075 cm |

The compact export check uses `--compact-tolerance 1e-4` instead of 1e-5.
- The fold/full difference was 3.05e-5, and only on the σ=10 synthetic probes, whose outputs
  are in the hundreds (float32 rounding).
- On all 10,388 real validation inputs it is 5.2e-6.
- The uniform compact was re-exported from its checkpoint with `scripts/export_compact_from_run.py`.

On the held-out no-hover validation pairs, the predicted/true speed ratio goes from 0.52 to 0.84,
and translation RMSE goes from 1.24 to 1.07 cm. On the original hover-start validation pairs the
new model over-predicts (1.42, previously 0.74).

## Evaluation

See `evaluation.md` (from `scripts/summarize_eval.py`) and `LEO_RESULTS.md`.
