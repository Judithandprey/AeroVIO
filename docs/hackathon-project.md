# AeroVIO for a Physical AI application

## Project description

**AeroVIO — Compute-aware visual-inertial motion estimation**

AeroVIO is a compact RGB+IMU odometry prototype that estimates camera pose and
twist inside a constrained WebAssembly runtime. It combines a small learned
motion predictor with geometric and inertial checks, and evaluates accuracy
alongside measured inference cost. Controlled experiments compare an official
KLT+IMU baseline, image/IMU input variants, training-only image warping, and
causal startup gyro calibration. The work is evaluated on simulated flights;
it is a localization component for future embodied systems, with no robot
action controller or deployed world model claimed.

## Personal contribution wording

> I co-designed AeroVIO's experimental direction, proposing perspective-aware
> motion estimation and a preserved-baseline workflow to test whether images,
> warping constraints and rotation-focused training improve localization.
> I used AI-assisted development to implement and evaluate the prototype,
> keeping reproducible experiments and negative results alongside improvements.

This wording reflects the recorded user decisions and the AI-assisted execution
in this project. It does not claim that the user independently wrote every
implementation, invented the upstream network/baseline, or demonstrated a
physical robot. Individual code ownership beyond these records needs separate
confirmation before stronger wording such as "solely implemented" or "led all
model engineering" is used.

## Result wording and evidence

> On ten local public simulated flights, a fixed calibration variant reduced
> mean Sim(3)-aligned trajectory error by 56.8% and maximum per-frame compute
> fuel by 89.6% versus the official KLT+IMU teaching baseline. Median orientation
> RMSE was 12.20 degrees versus 13.25 degrees, and local guest-step latency was
> 1.24 ms versus 3.21 ms. A complete repeat reproduced every pose, twist and
> frame's fuel exactly.

These are measured project results for the **calibrated-gyro diagnostic**.
Aligned trajectory error improves in all ten episodes; rotation improves in
six, so do not imply uniform orientation gains. Its original training remains
the v1 model; the v4 experiment is a fixed inference ablation with no retraining.
The [v4 report](../experiments/gyro-bias-v4/README.md),
[actual reports](../experiments/gyro-bias-v4/comparison.md), and
[repeat checks](../experiments/gyro-bias-v4/output-verification.md) support these
numbers. Local timing is not sensor-to-output latency or a course-server score.

The strict validation rule retains the current model: calibration-only rotation
improved by 68.77% on the four simulated validation flights, but aligned ATE
increased by 0.02752%, failing the predefined no-regression condition. A public
diagnostic result cannot retroactively change that validation decision.

Compared with the existing network rather than the baseline, the candidate
improves aligned ATE and rotation but costs 16.7% more step time and increases
median unaligned position error by 1.55%. Its fitted scale is still inaccurate.
The report and portfolio should keep this tradeoff and the initial-hover
assumption accessible rather than describing the variant as an unqualified
replacement or real-world-ready navigation system.

## Claims to avoid

- "Real-robot deployment", "robot action model", "world model", or reliable
  unseen-surface 3D reconstruction: these are not demonstrated by this project.
- "State of the art" or superiority to production VIO systems: the reference
  is the course's official teaching baseline under a pinned local runtime.
- Unqualified "metric map accuracy": Sim(3) alignment removes global scale,
  rotation and translation; unaligned errors and fitted scale must stay visible.
- Strong visual-rotation claims: the image-shuffle probe did not establish that
  correct visual correspondence drove the synthetic rotation gain.
- Statistical robustness from a deterministic repeat: repeated outputs verify
  reproducibility, while independent startup motions and physical sensors still
  need testing. The current generated validation flights share startup dynamics.

Source ownership and upstream attribution are in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md); the evidence and personal
boundaries are recorded in [portfolio-claims.json](portfolio-claims.json).
