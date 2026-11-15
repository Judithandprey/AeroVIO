# Frozen-predictor memory gate V12

Start from V11 commit `96e6fcb` on `experiment/memory-gate-v12`. Preserve all
previous models, artifacts, branches and workspace defaults.

Only a scalar trust gate is learned. The two predictors are frozen V10 and
fixed-final V11 memory. Translation is `(1-g)*V10 + g*V11`. The previous blended
pre-visual velocity is transported and clipped exactly as in V11. The gate
receives normalized state18 plus each candidate's three velocity components,
clipped to +/-2 m/s: 24 -> 8 -> 1 -> sigmoid, 209 trainable parameters and 200
semantic MACs. Both frozen networks plus gate total 3,165 parameters and 3,016
semantic MACs. Whole-estimator time/fuel must be measured, not inferred from MACs.

A fixed-mixture control learns only one scalar sigmoid bias. Adaptive and
constant arms start at g=0.5; the adaptive gate has zero output weights initially.
An exact adaptive training repeat is required. The two original predictor
parameters must remain bit-identical throughout training. Their input/output
normalization is unchanged. The raw rotation residual used for the photometric
loss remains V10's; deployed rotation remains the original calibrated gyro.

Training keeps the V10 target plan, existing data, 20-frame causal contexts,
self-predicted memory, original pose plus 2x warping loss, seed, optimizer,
40 epochs and 1,520 updates. No new data balancing, stop loss, augmentation,
visual measurements, zero-velocity update or stationary classifier is introduced.
This first gate can use inertial dynamics and predictor disagreement but has
no independent visual stationary cue. It cannot resolve every stationary versus
constant-velocity ambiguity.

Select each experimental candidate by full original-validation pose loss among
updated epochs 1..40. Record initial and hard-endpoint reference losses; this is
candidate selection, not permission to replace V10. Save final weights for
provenance, not a later long-score checkpoint sweep. Freeze all selected models
before reading long-flight estimator scores.

Compare V10, V11, learned constant mixture and adaptive gate on the six prior
V11 long flights and six fresh 60-second flights (seeds 440000 onward). Exact
original generator and successful-flight policy are reused. Report actual
coverage, including empty bins; do not select flights using estimator scores.
The prior set is development/regression evidence. New seeds are excluded from
training and checkpoint selection, but scene identities remain 5/6/7.

Keep existing stationary/turning/stop metrics. Define cruise before evaluation
using one-second true-motion windows: mean speed >=0.1 m/s, speed CV <=0.1,
speed range <=0.05 m/s, net displacement/path length >=0.95, mean angular rate
<15 deg/s. Report cruise length RMSE, bias, and predicted false-stationary runs
(five consecutive output increments below 0.02 m/s while true speed >=0.1).
Stops use the existing >=0.5 s below 0.02 after recent >=0.1 m/s motion rule.
These ground-truth classifications are evaluation labels, never runtime inputs.
Fewer than five fresh stopping events means the stopping guard is inconclusive.

Time all four artifacts with every execution order balanced in 24-frame blocks.
Exclude timing while simulation, training or builds run. Run endpoint identity
checks and independent adaptive repetition outside the timed interleave. A
logging guest verifies normalized gate inputs, gate outputs, motion and history
transport against Python/ONNX. It is excluded from timing.

The native vision and rotation source files remain unchanged. Only inference,
weights/export plumbing and diagnostic logging change. No main merge or course
server upload is part of this experiment.
