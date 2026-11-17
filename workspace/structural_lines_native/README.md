# Isolated structural-line rotation experiment

This source copies the selected V8 native pipeline and adds structural_lines.h
and structural_line_gate.h through solution.cpp. The original visual_native
source directory and checkpoint stay unchanged.

LINE_MODE=0 disables lines, 1 performs detection and counterfactual decisions
without changing outputs, and 2 enables bounded fallback corrections. V8
confidence weighting stays off and rotation refinement stays on in all builds.
Use scripts/build_structural_lines.py with a fresh attempt name. Each build
receives an ABI check; the disabled build also runs actual-WASM geometry tests.

The ABI remains 320x240 RGB. Detection uses 160x120 rounded 2x2 green-channel
averages and pixel-center-adjusted intrinsics. Raw gyro/tracker coordinates stay
unchanged. Corrections adjust output alignment and subsequent displacement
increments, with twist recomputed from the emitted poses.

line_diagnostics exports 21 doubles in the configuration's fixed order.
line_observation exports 128 doubles: eight header values, 24 line endpoint
records of four values, and three direction records of eight values. Both
exports are read-only and called outside timed step calls. A direction is not
a semantic wall, known global axis, metric width or scale measurement.

See docs/structural-lines-v13.md and experiments/structural-lines-v13/README.md
for the protocol and measured outcomes.
