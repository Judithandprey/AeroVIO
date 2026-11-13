# Training distribution contrast on the teacher estimator

This experiment tests the user's requested sampling change while preserving
teacher commit `0c4f7781cfc844d7fdc4f7223619a739b9f30cc7`. The existing
`experiment/native-fusion-v8` branch contains the work; the V8 inference options
are not included in this comparison. The original teacher WASM, all old weights,
and workspace inference defaults remain intact.

## Question and fixed settings

Does balancing startup/transition, approximately steady cruise, and turning
examples improve displacement magnitude without changing the network or loss?

The teacher uses the compact export of the earlier IMU-only training run. Both
new fits therefore use the same original zero-image training graph, then the
same algebraic export to 1,350 deployed parameters. Training the compact MLP
directly would change the optimization parameterization, so this experiment
does not do that.

`configs/data-balance-v9.json` fixes the thresholds, seed, optimizer settings,
validation rule, data and evaluation limits before fitting. There are 2,388
examples per epoch, batch size 64, 38 optimizer steps per epoch and 40 epochs.
Both arms retain seed 17, Adam with learning rate 0.001, the cosine schedule,
warping weight 2, CPU four threads, and unchanged preprocessing/normalization.
No GPU or new synthetic trajectories are needed for this contrast.

## Training-only categories

Categories have explicit precedence, so each frame pair belongs to exactly one:

1. The first second belongs to startup/transition.
2. Afterwards, true camera angular speed of at least 15 degrees/second is turn.
3. Otherwise, speed below 0.1 m/s or absolute tangential acceleration above
   0.2 m/s² belongs to startup/transition.
4. Remaining examples are approximately steady cruise.

Tangential acceleration is the time derivative of a five-frame moving average
of scalar speed. Cruise does not mean exactly zero 3D acceleration. Ground truth
only supplies these offline sampling labels, never network input or inference.

Original group counts are 1,346 / 749 / 293. Balanced sampling draws 796 from
each group per epoch, cycling randomly shuffled group members when necessary,
then shuffles the combined epoch. It changes exposure, not unique dataset size.
The control retains the original `torch.randperm` sequence. The unchanged
validation distribution selects each checkpoint by normalized pose loss.

## Evaluation and evidence

The control must reproduce the archived teacher ONNX exactly. Both compact
exports are checked against the full PyTorch network on all 796 validation
examples plus 64 synthetic probes. A full repeated balanced fit must reproduce
weights and validation metrics before runtime evaluation.

`scripts/run_data_balance.py` retrieves every native C++ file directly from the
teacher commit. Only build paths selecting each checkpoint change. Both rebuilt
WASMs undergo 80-frame checks and run on the original four scene-disjoint
validation flights. Control poses, twists and per-frame fuel must exactly match
the preserved teacher artifact. Balanced rotation must be unchanged.

Validation retains the teacher unless balancing lowers mean one-second
displacement-length RMSE by at least 2%, does not worsen mean Sim(3) ATE, retains
exact rotations and stays within a 5% peak-fuel guard. The selection is frozen
before public evaluation. Both variants are reported regardless of selection.

Five serial public evaluations cover teacher, rebuilt control, balanced,
balanced repeat and teacher repeat. The complete ten-episode sets and resource
limits match earlier experiments. Repeats verify exact pose/twist/accuracy/fuel;
wall-clock time is compared as a host-dependent measurement. Training stops
before these timings are collected.

Serial timing varied substantially even between byte-identical teacher/control
artifacts. A timing-only follow-up, `scripts/time_data_balance.py`, alternates
which of the two fixed WASMs executes first on each identical public frame.
It rechecks poses, twists and fuel against completed runs. This follow-up changes
no weights, selection rules, inference settings or accuracy results.

Read `experiments/data-balance-v9/README.md` for results and interpretation.
The previously inspected public scene is a diagnostic comparison, not a fresh
independent test set. A same-seed repeat verifies determinism, not across-seed
significance. No server submission or workspace model-default change is made.

## Reproduction

Use a fresh checkout/output tree with the pinned upstream installation and
preserved `data/motion-cache-v2`, `data/hyperdrone-motion-v2`, teacher WASM and
native build dependencies. Existing evidence directories are not overwritten.

```bash
.venv/bin/python scripts/prepare_motion_balance.py
.train-venv/bin/python -m unittest discover -s tests -v

.train-venv/bin/python scripts/train_motion.py \
  --cache data/motion-cache-v2 --out models/balance-v9-control \
  --compact-out models/balance-v9-control-compact --input-mode imu_only \
  --warp-weight 2 --epochs 40 --batch 64 --lr 0.001 --seed 17 --threads 4

.train-venv/bin/python scripts/train_motion.py \
  --cache data/motion-cache-v2 --out models/balance-v9-balanced \
  --compact-out models/balance-v9-balanced-compact --input-mode imu_only \
  --warp-weight 2 --epochs 40 --batch 64 --lr 0.001 --seed 17 --threads 4 \
  --sampling-plan data/motion-balance-v9/balanced-plan.npz

.train-venv/bin/python scripts/train_motion.py \
  --cache data/motion-cache-v2 --out models/balance-v9-balanced-repeat \
  --compact-out models/balance-v9-balanced-repeat-compact --input-mode imu_only \
  --warp-weight 2 --epochs 40 --batch 64 --lr 0.001 --seed 17 --threads 4 \
  --sampling-plan data/motion-balance-v9/balanced-plan.npz

mkdir -p results/data-balance-v9
.venv/bin/python scripts/run_data_balance.py --phase validation
.venv/bin/python scripts/run_data_balance.py --phase public
.venv/bin/python scripts/time_data_balance.py
.venv/bin/python scripts/report_data_balance.py
```
