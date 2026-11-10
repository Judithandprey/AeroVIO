# Perspective geometry and sparse spatial memory

**Status:** implemented as the optional map stage of the learned-motion pilot.
The default working model has map correction disabled. The original KLT + IMU
reference is preserved at tag `baseline-klt-imu-v1`.

Read [learned-motion.md](learned-motion.md) for the trained network, warping loss,
data separation and runtime architecture. The [controlled comparison](../experiments/learned-motion-v1/comparison.md)
contains measured outcomes; implemented geometry is not itself evidence of an
accuracy improvement.

## Research question and observed boundary

Can a bounded, causal landmark map improve the accuracy-compute trade-off of a
small learned visual-inertial motion estimator?

The initial map-enabled pilot increased mean Sim(3) positional error and runtime
relative to the same warp-trained weights with the map disabled, while reducing
rotation error. The map remains an opt-in experiment. See the complete comparison
for individual episodes, scale, twist and failure metrics.

Perspective projection, triangulation, KLT and PnP are established methods.
The original KLT + IMU reference already uses camera intrinsics and two-view
`findEssentialMat`/`recoverPose`; it does not maintain a persistent 3D map.
The experiment tests a persistent local geometry correction on a learned prior.

## Projection and coordinate convention

Let W be the first camera frame. `T_WC` maps current camera coordinates into W.
For a stored landmark `P_W`, compute `P_C = inverse(T_WC) P_W`. With
`P_C = (Xc, Yc, Zc)` and `Zc > 0`:

```text
u = fx * Xc / Zc + cx
v = fy * Yc / Zc + cy
```

Use each episode's intrinsics. Images are rectified; no lens distortion is added.
`T_cam_imu` maps IMU to camera. OpenCV PnP returns `T_CW`; invert it before updating
`T_WC`. The output quaternion is scalar-last. Twist is expressed at the current
camera centre in current camera axes and is recomputed after any map correction.

Perspective describes image projection; parallax supplies depth constraints.
Raw pixel motion includes rotation. Triangulation uses rotation-compensated
bearing change and rejects weak baselines, behind-camera points and poor
reprojection consistency. A consistent projection alone does not resolve
monocular metric scale.

## Implemented data flow

```text
RGB pair + IMU + intrinsics -> small CNN -> predicted relative camera motion
previous corrected pose + predicted motion -> current pose prior
RGB -> persistent KLT tracks + bounded keyframes -> sparse 3D landmarks
existing landmarks + current observations -> PnP/RANSAC -> gated correction
accepted correction or unchanged learned prior -> pose and camera-axis twist
```

Only earlier/current observations and estimated poses enter the map. Newly
triangulated points cannot correct the frame that created them. No ground truth,
prebuilt evaluation-scene map, future observations, loop closure or rewriting of
previous outputs is used. All state resets at episode initialization.

Static scene geometry is the working assumption. Landmarks remain in this local
map while their image tracks remain valid; lost or occluded tracks are discarded.
There is no landmark re-identification after disappearance. A never-observed
surface becomes a candidate after it is observed and can be triangulated reliably.
Dense reconstruction and generative world/action models are outside this
implementation.

## Fixed pilot budgets and gates

- At most 160 active tracks/landmarks and four retained anchor keyframes.
- KLT forward/backward agreement; 15x15 windows, two pyramid levels and
  a 12-iteration tracking cap.
- Keyframe interval 5; minimum 0.03 m baseline and 1.5-degree compensated parallax.
- Positive depth in the 0.2-30 m range and two-view reprojection checks.
- At least 12 map correspondences; PnP RANSAC capped at 40 iterations and
  refinement at five iterations, with a 2-pixel reprojection threshold.
- Reject large rotation/translation innovations and geometric degeneracy.
  Rejection retains the learned pose prior exactly.

These settings were fixed for the pilot rather than tuned on public results.
A map built from poor estimated poses can reinforce its own scale or drift errors.
Bounded computation and conservative gates do not guarantee beneficial corrections.

## Source and tests

- `workspace/solution/solution.py`: lifecycle, learned motion integration and ABI.
- `workspace/solution/motion_inference.py`: ONNX model and metadata validation.
- `workspace/solution/preintegrate.py`: shared training/inference IMU features.
- `workspace/solution/local_map.py`: tracking, triangulation, bounded map and PnP.
- `training/motion_model.py`: host-only network and differentiable warping objective.
- `tests/test_local_map.py`: coordinate, triangulation, rejection and tracking checks.
- `tests/test_motion_runtime.py`: state composition, camera-axis twist and map integration.
- `tests/test_warp_training.py`: training projection, gradients, visibility and ONNX parity.

The map's triangulation, SVD, PnP and refinement operations were exercised in
WASM, and the complete learned/map variant passed a 20-frame synthetic check.
Full public evaluation remains the evidence for trajectory behavior and compute.

## Controlled comparisons

B0 is the frozen upstream KLT baseline. N1 and N2 share architecture, training
data, initialization, batch order and schedule; only the warping loss changes.
N3 uses N2's identical weights with map correction enabled. This separates the
training-loss effect from the inference-time map effect. B0 versus N3 alone
would mix several changes.

Inspect mean Sim(3) ATE, every episode, maximum fuel, failed/filled frames,
unaligned ATE, rotation and twist errors. Sim(3) removes a global scale factor.
This single-seed, small-data pilot does not establish unseen-scene or physical
robot performance. Follow [setup](setup.md) to package variants explicitly.

## Primary references

- [OpenCV projection and calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [OpenCV PnP and refinement](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
- [Pinned KLT + IMU source](https://github.com/jonas-eschmann/viobench/blob/2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca/viobench/solution/examples/python/klt_imu/solution.py)
- [Pinned tiny-CNN teaching example](https://github.com/jonas-eschmann/viobench/tree/2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca/viobench/solution/examples/python/tiny_cnn)
