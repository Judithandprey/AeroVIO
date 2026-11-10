# Learning how rotation changes a view

The proposed experiment trains the existing motion network on more diverse
rotational view changes. It does not add a larger deployed network. Its matched
control continues training on the original mixed-motion data for the same
number of optimizer steps; the experimental variant adds a rotation-only
auxiliary objective.

In code and logs, `real` means the original simulated flight sequences containing
translation and rotation, as opposed to the extra pure-rotation image pairs.
Neither split is physical-robot or real-camera data.

## Why pure rotation gives a useful geometric lesson

For a fixed optical centre, let `R_2_from_1` transform 3D coordinates from camera
view 1 into view 2. With camera calibration matrices `K1` and `K2`, the pixels
obey the homogeneous relation:

```text
p2 ~ K2 @ R_2_from_1 @ inverse(K1) @ p1
```

The unknown point depth cancels. This works for a general static 3D scene; it
does not require every point to lie on one plane. The fixed-optical-centre
condition matters: a rotating robot may also translate its camera, in which
case full motion/depth geometry is needed.

Chinese hints: optical centre = 光心; pure rotation = 纯旋转;
homography = 单应变换; rotation residual = IMU 旋转估计的修正量.

| Rotation in camera coordinates | Typical view change |
| --- | --- |
| Around camera x, the right axis | Pitch-like change; points sweep mainly vertically |
| Around camera y, the down axis | Yaw-like change; points sweep mainly horizontally |
| Around camera z, the forward axis | Roll-like change around the principal point |
| Combined axes | A full perspective transformation |

Yaw and pitch are not equivalent to simply rotating a flat picture in 2D.
The experiment uses the full calibrated projective mapping. Axis descriptions
are for the camera convention, not a claim about the drone's world yaw/pitch/roll.

## What is added to training

Source images come only from the existing HyperDrone training and validation
scenes. Each source image supplies two virtual views with symmetric minus/plus
half-angle rotations about the same optical centre. Both views use the same
interpolation and a fixed central crop. Pairs whose rays would leave the source
image are rejected, avoiding an angle-dependent black-border shortcut.

The images first follow the existing 320x240 adaptation, are reprojected at that
resolution, then use the deployed grayscale/area-resize preprocessing to 64x48.
Training balances camera x/y/z and mixed-axis rotations over 0-3 degrees per
0.05-second interval. All geometry, source identities, seeds and checks are saved.

The synthetic IMU must describe the same virtual motion. It uses source-episode
gravity, camera/IMU extrinsics and sensor-noise/bias priors. For a nonzero lever
arm, constant angular motion also produces centripetal acceleration at the IMU.
The ten 200 Hz samples pass through the same preintegration helper as inference.
No unrelated real translational IMU window is attached to a rotated image pair.

The auxiliary loss supervises **rotation only**, comparing the complete predicted
rotation `R_imu @ R_residual` with the known virtual-view rotation on SO(3).
It does not train the translation head to output zero for these samples. The
original real-data pose and photometric losses remain active, including actual
translation/rotation mixtures.

## Keeping the experiment interpretable

Both fine-tunes start from exactly the v1 warp-trained checkpoint with frozen
normalization. They share real minibatches, initialization, optimizer restart,
learning-rate schedule and optimizer-step count. The auxiliary sampler has its
own random generator. Extra auxiliary training compute is reported separately.

Checkpoint selection uses accumulated rotation error on **real held-out**
validation flights, with guards on both pairwise translation error and integrated
trajectory-position error. Epoch zero is a valid candidate: if further training
does not improve within the guards, the initial checkpoint is retained.
Synthetic validation is diagnostic only. Public trajectories do not select
checkpoints, weights, crop factors or noise settings.

A separate image-pair-shuffling probe asks whether predictions depend on visual
information. It keeps states and labels fixed and permutes whole image pairs
with an independent seed. This is an out-of-distribution sensitivity test; a
change under shuffled inputs is not by itself proof of useful visual inference.
Correctly paired real validation and complete public trajectories remain needed.

Pure-rotation data also cannot by itself teach stable long-term bias correction,
or establish performance during arbitrary translation and rotation. The model's
small image resolution and global pooling may limit what it can infer. The
experiment measures those limits instead of assuming more training must help.

## References and records

- [OpenCV's pure-rotation homography derivation](https://docs.opencv.org/4.11.0/d9/dab/tutorial_homography.html#tutorial_homography_Demo5).
- [Fixed experiment protocol](../configs/rotation-focus-v3.json).
- [Completed experiment record](../experiments/rotation-focus-v3/README.md).
- [Warping and world models](warping-and-world-models.md).
