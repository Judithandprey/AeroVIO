# Explicit correspondence supervision v6

The fixed matching-supervision trial is complete. It does **not** establish a
matching or rotation upgrade over V5 temporal. Supervised cross-entropy improves,
but subpixel correspondence endpoint error worsens on both training and
validation data. Public median rotation error also worsens. The new weights are
retained as unselected diagnostics; all previous versions and selection decisions
remain unchanged.

See [public comparison](comparison.md), [figure](comparison.png),
[matching audit](matching-audit.json), [runtime validation](validation.json),
[frozen decision](selection.json), [protocol](../../configs/correspondence-v6.json)
and [geometry/reproduction guide](../../docs/correspondence-supervision.md).

## What changed

Both new models begin from the exact V5 temporal epoch-7 checkpoint and use the
same 35,641-parameter network. The 29,398-parameter translation base stays frozen;
the 6,243 encoder/head parameters continue training. The new **supervised** arm
adds `0.1 * soft-label matching cross-entropy` to the existing rotation objective.
The **continuation** arm receives only the original rotation objective.

Labels project current camera-z depth through the true relative camera pose into
the previous image. Bilinear probability mass over the nine local offsets keeps
subpixel displacement. Fixed geometric visibility, positive depth, search-range
and full-neighborhood masks exclude unsupported supervision. The actual
convolution lattice is used, rather than resized-image centre coordinates.
Reference geometry never enters the deployed estimator.

Both fits use seed 17, 40 epochs, 1,520 optimizer steps, 95,520 pair exposures,
identical starting weights/orders, fresh Adam and learning rate 1e-4 to 1e-5.
Scenes 2/3/4 train; scenes 5/6 validate. Public scene 1 is excluded. No settings
were tuned on public results. Model selection uses minimum original-flight
validation rotation error among trained epochs; the starting V5 artifact remains
separate. The new selected trained epochs are 4 (continuation) and 13 (supervised).

## Direct matching results

Endpoint error is measured in **64x48 input-image pixels**, with equal weighting
of episodes. The moving subset requires true displacement of at least 0.25 pixel.

| Descriptor/control | Training EPE | Validation EPE | Validation moving-subset EPE |
| --- | ---: | ---: | ---: |
| Always zero displacement | 0.267836 | 0.236517 | 0.609776 |
| V5 temporal | 0.355287 | 0.336266 | 0.677546 |
| V5 single-frame, actual current/current volume | 0.981860 | 0.995922 | 1.158145 |
| V6 continuation | 0.330358 | 0.306630 | 0.660744 |
| V6 supervised | 1.084419 | 1.067577 | 1.251686 |

Matching did not become more accurate under these displacement metrics. The
training matching cross-entropy drops from 2.1493 in epoch 1 to 1.2809 in epoch 40;
validation cross-entropy also drops (2.1232 to 1.3569), while endpoint error
worsens. Thus decreasing this classification objective did not ensure accurate
continuous displacement. The selected epoch-13 validation EPE is 1.0676; the
last epoch's EPE is 1.0313, also worse than the starting model.

Motion is small: 99.56% of training points and 99.95% of validation points round
to the centre candidate. A high nearest-cell accuracy would therefore be weak
evidence. The zero-displacement control outperforms the learned decoders here,
including on the fixed moving subset. These observations constrain the claim;
they do not prove one unique failure mechanism.

The independent OpenCV projection audit checks 17,607 training and 17,541
validation points across 128 sampled pairs in each split. Maximum discrepancies
are 0.00001045 and 0.00000969 pixels, within the fixed 0.0005-pixel tolerance.
Coordinate, sign, subpixel mass and mask unit tests also pass. The training-set
EPE regression shows that the effect is not confined to unseen validation scenes.

## Original-flight validation and selection

| Variant | Mean rotation RMSE (deg) | Mean Sim(3) ATE (m) | Mean unaligned ATE (m) |
| --- | ---: | ---: | ---: |
| Calibrated gyro reference | 2.170470 | 0.265630 | 0.511332 |
| V5 single-frame | 2.453287 | 0.265527 | 0.512153 |
| V5 temporal | 2.780753 | 0.265555 | 0.513904 |
| V6 continuation | 3.000643 | 0.265556 | 0.514282 |
| V6 supervised | 2.727595 | 0.265563 | 0.511603 |

Supervised fine-tuning slightly improves rotation against starting V5 temporal,
but both new candidates fail the predeclared pose conditions against calibrated
gyro and improve rotation on 0/4 of those reference flights. The supervised arm
also fails all four matching-gain conditions against V5 and continuation, for
all-valid and moving-subset EPE. Neither matching nor pose thresholds are relaxed.
V6 retains previous choices; V5's calibrated-gyro reference, V4's selected
`current`, and the workspace pose-only default keep their original roles.

## Local public comparison

All seven serial runs complete 10/10 episodes (12,000 frames each), with no
failures or filled outputs. Values are from this round's measurements.

| Variant | Mean Sim(3) ATE (m) | Median rotation RMSE (deg) | Median WASM step (ms) | Max fuel/frame |
| --- | ---: | ---: | ---: | ---: |
| Official baseline | 2.041635 | 13.2449 | 3.102 | 141,910,846 |
| Calibrated gyro | 0.881818 | 12.1963 | 1.212 | 14,757,111 |
| V5 single-frame | 0.855547 | 9.0634 | 1.456 | 20,141,128 |
| V5 temporal | 0.870938 | 9.7654 | 1.561 | 21,854,083 |
| V6 continuation | 0.864435 | 10.6038 | 1.516 | 21,874,382 |
| V6 supervised | 0.868606 | 10.6784 | 1.506 | 21,874,382 |
| V6 supervised repeat | 0.868606 | 10.6784 | 1.517 | 21,874,382 |

Against V5 temporal, supervision reduces mean aligned ATE by only 0.27% and
increases median rotation RMSE by 9.35%. Mean episode rotation RMSE also increases
(12.7369 to 12.8759 degrees). Rotation wins occur in 6/10 episodes; the complete
per-episode table is retained, so aggregate regression is not a claim that every
episode gets worse. Against matched continuation, supervised rotation is 0.70%
worse and aligned ATE is 0.48% worse. It does not beat the V5 single-frame control.

The deployed V5/V6 ONNX graph structure is identical after removing initializer
values, and guest code is unchanged. Maximum fuel differs by only +0.093%
versus V5 temporal and is equal across the two V6 candidates. Local step times
are close; their small differences are not evidence of a new computational
speedup. The broad baseline advantage includes earlier work and cannot be
attributed to this matching supervision.

Sim(3) fits scale; median fitted scale remains about 2.14 for the learned
candidates. Step time excludes complete sensor-to-output latency. This is one
training seed, four short validation flights and one public scene, with no
physical-robot or held-out-server result.

## Verification and deliverables

118 host tests pass, including seven new geometry/gradient tests. Both new WASM
artifacts pass 40-frame checks. Every validation sample passes ONNX/OpenCV parity;
actual `Solution.step` rollouts confirm raw input/causal-prior identity, fixed
translation and agreement with training-side metrics. The V5 validation outputs
repeat exactly. The complete supervised public repeat has identical pose, twist,
accuracy and per-frame fuel. Four preserved public references reproduce their
previous results exactly. See [output verification](output-verification.json).

`continuation/` and `supervised/` retain ONNX, metadata, complete training histories
and visual probes. `runs/` retains manifests/reports/logs; cache provenance and
frozen selection are alongside them. Large checkpoints/data/raw trajectories stay
in ignored `models/`, `data/` and `results/`. No earlier experiment is overwritten.

Verified Windows copies are in
`C:/Users/ROG/Documents/GitHub/AeroVIO/artifacts/`:
`AeroVIO-correspondence-supervised-v6.wasm` and
`AeroVIO-correspondence-control-v6.wasm`, with SHA-256 sidecars and
`windows-artifacts.json`. They are unselected diagnostic candidates and have not
been submitted to the course server.

A next isolated hypothesis would optimize continuous displacement directly and
use more visibly moving examples, then check the zero-motion baseline and pose
errors again. This has not been implemented or demonstrated as an improvement.
