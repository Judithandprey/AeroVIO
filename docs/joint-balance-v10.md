# Joint speed and turn sampling

V9 balanced three motion categories, but most turning examples were slow.
Consequently its low-speed exposure rose rather than fell. V10 first fixes
the speed marginal, then balances turn rates within each speed bin subject to
support and per-example reuse limits. The network and all other training
settings are retained.

## Frozen treatment

`configs/joint-balance-v10.json` defines four speed bins, four turn-rate bins,
597 draws from each speed bin per epoch, and a maximum of four exposures for any
one example in an epoch. Within a speed bin, integer quotas are allocated as
equally as feasible across nonempty turn cells, capped at four times unique
support. Empty cells receive no quota. This avoids giving a single rare example
hundreds of exposures.

The pre-training audit verifies every epoch, including exact 25% low-speed
exposure, 2,388 examples, the per-example cap, all cache hashes and complete
sample coverage over 40 epochs. It is saved before training starts. Ground-truth
speed and angular rate choose examples only; they are never predictor inputs.

The original cache, normalization, seed 17, 40 epochs, batch 64, optimizer,
schedule, loss, preprocessing and original short-validation checkpoint rule are
unchanged. Training retains the original zero-image graph before equivalent
folding into 1,350 deployed parameters. A same-seed repeat verifies determinism.

## New long validation

`configs/long-validation-v10.json` fixes six fresh 60-second flights, two each in
scenes 5/6/7, starting with seed 420000. Scenes 5/6 were previously used for short
validation; scene 7 is new. All three are excluded from training. Model scores
never determine which flights are accepted; non-crash acceptance and retry limits
are predefined and all attempts are retained.

The existing simulator header and successful-flight capture function are reused.
Raw data resides on D: via a new `data/long-validation-v10` symbolic link in the
WSL checkout because the host C: drive has limited free space. No earlier data
is moved or deleted. The reused capture metadata's `training_only` field denotes
non-public generated data; these new flights are evaluation-only and never enter
the trainer or normalization calculation.

Checkpoints and build identities are frozen before first inference on these
flights. Three artifacts—teacher Native 160, V9 category-balanced and V10 joint—
plus a full V10 repeat run in rotating order on identical frames. Generation and
training finish before timing. The repeat must match poses, twists and fuel;
all models must preserve teacher orientations and angular twists.

The initial four-way run includes a repeated V10 artifact and showed cache
sensitivity. A final timing-only pass uses three distinct artifacts, shuffling
all six orders in each six-frame block. Each model occupies each execution
position exactly 400 times per flight. Outputs and per-frame fuel must reproduce
the accuracy run exactly. This follow-up changes no models or accuracy criteria.

Read `experiments/joint-balance-v10/README.md` for the finished result. Selection
uses the fixed distance/trajectory/fuel guards in the protocol; it does not
change a workspace default or authorize a course-server upload.

## Reproduce in a fresh output tree

Preserve existing output directories. Use the pinned dependencies, original
training cache and raw flights, V9 sampling labels/plans, teacher and V9 WASMs,
and native build tools. Adjust the physical data destination only in a new
predeclared protocol when reproducing on a different machine.

```bash
mkdir -p results/joint-balance-v10
.venv/bin/python scripts/prepare_joint_sampling.py
.train-venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/train_joint_balance.py
.venv/bin/python scripts/generate_long_validation.py
.venv/bin/python scripts/run_joint_balance.py --phase build
.venv/bin/python scripts/run_joint_balance.py --phase evaluate
.venv/bin/python scripts/time_joint_balance.py
.venv/bin/python scripts/report_joint_balance.py
```

For an exact reproduction, retain the host-test log at
`results/joint-balance-v10/host-tests.log`, as the build audit archives it.
Training and simulation may overlap; inference timing must start after both
finish. All scripts refuse to overwrite their primary run directories.
