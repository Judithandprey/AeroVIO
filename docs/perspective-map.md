# Perspective-aware spatial memory

**Status: design only.** This branch starts from the measured, unchanged official KLT + IMU baseline. The proposed map estimator has not been implemented or evaluated.

## Research question

Can a small, causal 3D landmark map reduce localization drift at an acceptable increase in maximum per-frame WASM fuel?

The baseline already uses perspective and epipolar geometry through camera intrinsics, `findEssentialMat`, and `recoverPose`. It detects new corners each frame and has no persistent 3D landmarks or bundle adjustment. The proposed contribution is persistent geometric consistency with bounded computation; perspective projection, triangulation, and PnP are established methods.

## Projection and coordinates

Let W be the first camera frame. Store camera pose `T_WC`, which maps current camera coordinates to W. For a landmark `P_W`, compute `P_C = inverse(T_WC) P_W`. With `P_C = (Xc, Yc, Zc)` and `Zc > 0`:

```text
u = fx * Xc / Zc + cx
v = fy * Yc / Zc + cy
```

Use the episode's intrinsics. Images are rectified; do not add lens distortion. `T_cam_imu` maps IMU to camera. OpenCV PnP returns the map-to-camera transform `T_CW`; invert it before publishing `T_WC`. Output twist is at the current camera centre in current camera axes.

Perspective describes projection; parallax is the viewpoint change that provides depth information. Raw pixel motion includes rotation, so use rotation-compensated bearing change to select triangulation candidates. Pure rotation, tiny baseline, weak texture, and repetitive structure can make depth unreliable. Near-large/far-small appearance in one image does not establish absolute depth or metric scale.

## Proposed data flow

```text
RGB -> persistent KLT tracks -> keyframes -> sparse 3D landmarks
IMU -> rotation/translation prediction ------------|
map points + current 2D observations -> PnP/RANSAC -> gated pose update
accepted pose + IMU state -> ABI pose/twist; rejected update -> IMU fallback
```

Triangulate from earlier/current observations and estimated poses only. Ground truth stays in evaluation. Do not prebuild a map of evaluation trajectories or revise outputs already returned. Reset all state between episodes.

The map is explicit numerical state. Start with sparse reliable points; dense point clouds and training a world model are outside this first experiment. PyTorch is unnecessary for this geometric stage.

## Incremental comparisons

| Stage | Change | Question |
| --- | --- | --- |
| B0 | Unchanged official KLT + IMU | What accuracy, failures, and fuel do we start with? |
| P1 | Persistent tracks and triangulation; no map feedback | Are map points geometrically consistent? |
| P2 | Gated 2D-3D PnP/reprojection pose correction | Does map feedback reduce drift? |
| P3 | Explicit feature, keyframe, landmark and iteration budgets | Can we retain the gain with less worst-frame fuel? |

For P2, compare map feedback enabled/disabled with otherwise identical code. B0 versus P2 alone mixes tracking changes with map feedback. Log inlier counts, rejected updates, landmark counts, reprojection errors and depth/parallax checks. Low reprojection error alone does not prove correct scale or lower trajectory error.

Initial, untested settings: at most 300 tracks, 5 recent keyframes, and 500 landmarks, with explicit RANSAC iteration caps. Keep position, rotation, velocity and camera/IMU transforms consistent when accepting a correction. Reject non-finite results, behind-camera points, weak support and geometric degeneracy. Avoid hardcoded public-scene structure.

The measured baseline has substantial unaligned drift on several episodes. Triangulating from poor baseline poses can create a self-consistent but wrong map. Validate initialization and scale before trusting map feedback; retain the IMU fallback when geometry is uncertain.

## Proposed files

- `workspace/solution/solution.py`: lifecycle, IMU prediction, fusion, ABI.
- `workspace/solution/geometry.py`: projection, transforms, triangulation checks.
- `workspace/solution/tracking.py`: persistent track IDs, KLT, outlier rejection.
- `workspace/solution/local_map.py`: bounded keyframes, landmarks, observations.
- `workspace/solution/pose_update.py`: PnP and acceptance checks.
- `tests/test_geometry.py`: known-transform projection/PnP, extrinsic direction, pure-rotation rejection and degenerate correspondences.
- `experiments/<run>/`: parameters, provenance, reports and conclusions.

These modules and tests are planned. Keep host-only tests out of the bundled solution directory. Confirm required functions work in WASM OpenCV; host Python success alone is insufficient.

## Acceptance

Use the same ten public hashes, limits, runner and runtime. Inspect mean Sim(3) ATE and every episode, maximum fuel, failures/filled frames, rotation error, velocity error and unaligned ATE. Sim(3) alignment removes a global scale factor: a good aligned score alone does not establish a metrically accurate map.

Report regressions and repeatability. Public-set improvements do not prove unseen-scene performance, physical robot success or hackathon impact.

## Primary references

- [OpenCV projection and calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [OpenCV PnP and reprojection refinement](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
- [Pinned baseline source](https://github.com/jonas-eschmann/viobench/blob/2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca/viobench/solution/examples/python/klt_imu/solution.py)
