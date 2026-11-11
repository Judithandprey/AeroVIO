# Explicit correspondence supervision

V6 adds geometric matching labels during training of the existing V5 correlation
network. The deployed architecture, 35,641 total parameters, frozen translation
network, causal gyro calibration, input contract and guest Python source are
unchanged. The matching loss and all reference depth/pose data remain host-only.
See the [frozen protocol](../configs/correspondence-v6.json) and
[completed experiment record](../experiments/correspondence-v6/README.md).

## Geometric labels and masks

The original cache supplies current camera-z depth, pinhole intrinsics, true
relative camera motion and a visibility mask derived from true geometry.
For a current pixel `q`, unprojection yields `X_current = depth * inverse(K) * q`.
The label projects `X_previous = R_previous_current * X_current + t` into the
previous image. The desired displacement is **previous location minus current
location**, in camera-image coordinates.

Both 3x3, stride-2, padding-1 encoder convolutions preserve a known lattice:
the centre of feature `(x,y)` lies at image pixel `(4*x,4*y)`. This is different
from resizing an image to 16x12 with half-pixel centres. Labels explicitly use
the convolution lattice, with the existing low-resolution depth sampling
approximation (recorded 0.05-pixel offset) retained.

The 3x3 correlation channels enumerate `dy` then `dx`, each in `{-1,0,1}`.
Continuous displacement in feature cells distributes bilinear probability mass
over its neighboring offsets. This retains small motions: for example, image
displacement `(1,-1.2)` pixels becomes feature displacement `(0.25,-0.3)`, with
four nonzero label weights instead of a rounded centre label.

Only fixed, valid training geometry contributes. A point must pass the original
depth/visibility mask, have positive current and projected depth, remain within
the radius-1 search window, and have every candidate in its 3x3 neighborhood
inside the feature image. Occluded, disoccluded, invalid and unsupported points
are excluded. Neither a learned confidence nor a predicted mask chooses labels.

The training set has 329,148 valid points across 2,388 pairs; validation has
108,751 across 796 pairs. No pair is empty. Mean displacement is 0.268/0.237
low-resolution image pixels. Rounding displacement chooses the central cell for
99.56%/99.95% of training/validation points, so nearest-cell accuracy is an
uninformative primary metric here.

## Fixed experiment

Both new runs start from the exact V5 temporal epoch-7 checkpoint. A fresh Adam
optimizer updates the existing 6,243 encoder/head parameters for 40 epochs,
batch 64, seed 17, four CPU threads, cosine learning rate 1e-4 to 1e-5, and
gradient clipping at 5. The base network remains frozen. Initial weights and
all 1,520 optimizer-step minibatches match between runs.

- **Continuation:** unchanged calibrated-rotation Smooth L1 objective.
- **Supervised:** the same objective plus `0.1 * matching_cross_entropy`.

Matching cross-entropy uses `log_softmax(correlation / 0.1)` against the
bilinear soft labels. Each usable image pair contributes equally, averaging over
its valid points. Empty masks return finite zero matching loss; the rotation
objective remains active. The shared encoder is trained through the matching
loss directly, even if the rotation head is initially insensitive to its features.

Training scenes remain 2/3/4 and validation scenes 5/6. No public-scene training,
hyperparameter search, synthetic auxiliary augmentation, new warping objective
or multi-frame objective is added. Each run retains its trained epoch with lowest
mean episode accumulated validation rotation RMSE; earlier epochs win ties.
Epoch zero is preserved separately as the V5 artifact. No public result selects
weights or changes the acceptance rule.

The pose acceptance rule retains V5's engineering budget against calibrated gyro:
at least 5% lower mean rotation error, wins on at least three validation flights,
and no more than 1% aligned or unaligned position degradation. V6 additionally
requires at least 5% lower matching endpoint error than both V5 temporal and the
matched continuation, on all valid points and the moving subset. Earlier
selection decisions and the working pose-only default remain independent.

## Matching verification and diagnosis

For diagnosis, softmax probabilities over the nine candidate offsets yield an
expected displacement (soft argmax). Endpoint error is measured in **64x48 image
pixels**, averaging valid points within each episode and then averaging episodes.
It is not error at the 320x240 sensor-adapted resolution. A separate subset
requires true displacement of at least 0.25 image pixels. The zero-displacement
baseline is reported for both subsets, preventing a trivial central prediction
from being mistaken for useful motion matching.

V5 temporal, V5 single-frame's actual current/current volume, matched continuation
and supervised descriptors are evaluated on identical masks and labels. A
previous-image permutation diagnostic fixes the current image, IMU prior, pose
labels and translation; it is an out-of-distribution sensitivity test.

Seven new tests cover exact lattice centres, independent OpenCV projection,
signed/subpixel label mass, occlusion/search support, finite empty-mask gradients,
soft-argmax decoding and direct encoder gradients. An additional data audit checks
128 sampled pairs per split against independent OpenCV projection, then measures
matching on both train and validation splits. Actual `Solution.step` validation
checks causal priors, frozen translation and training/runtime scoring parity.

The seven public runs include exact preserved baseline, calibrated gyro, V5
single-frame and V5 temporal artifacts, both new candidates and a supervised
repeat. All runs are serial after fitting and matching audits stop. Report local
WASM step time and deterministic fuel separately; neither is complete system
latency. Sim(3) alignment removes scale.

## Reproduction and artifact identity

The recorded commands run from WSL `/root/projects/AeroVIO`. Existing output
directories are refused; preserve them when preparing a new reproduction.

```bash
.train-venv/bin/python scripts/train_correspondence.py --prepare
.train-venv/bin/python scripts/train_correspondence.py --variant continuation --out models/correspondence-continuation-v6
.train-venv/bin/python scripts/train_correspondence.py --variant supervised --out models/correspondence-supervised-v6
.train-venv/bin/python scripts/validate_correspondence.py
.train-venv/bin/python scripts/audit_correspondence.py
.venv/bin/python scripts/run_correspondence_evaluation.py
.train-venv/bin/python scripts/report_correspondence.py
```

Use new names when packaging archived diagnostics:

```bash
.venv/bin/python scripts/package_motion.py --model experiments/correspondence-v6/supervised --name correspondence-supervised-repro --calibrate-gyro
.venv/bin/python scripts/package_motion.py --model experiments/correspondence-v6/continuation --name correspondence-control-repro --calibrate-gyro
```

The new metadata identifies `training_stage: correspondence-v6-<variant>` and the
parent V5 model hash. The inference graph is still `local-correlation-v5`.
Building and measuring these artifacts does not submit them to the course server.
