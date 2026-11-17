# Build and test boundary

The experiment adds no neural parameters and changes no original V8 checkpoint,
point tracker, gyro integration or visual fusion source. Three primary evaluated
WASMs are identified in artifacts.json. A disabled-line rebuild additionally
checks output identity against the preserved V8 artifact.

Build a1 compiled successfully and passed the disabled-module ABI check. Its
whole-test invocation exhausted the old V8 helper's 200,000,000 fuel allocation
while exercising repeated synthetic image sequences. The original log is retained
under build-a1/geometry-tests.log. This was not an estimator per-frame failure.

The identical test binary passes with the new test_structural_lines.py batch
runner, which allows 2,000,000,000 fuel for the complete multi-image test suite.
All 40 checks pass and consume 409,336,583 fuel in total. No fixture, assertion,
gate threshold or estimator resource limit was relaxed. Build a2 uses that batch
runner and completes all three 80-frame ABI checks. The disabled solution WASMs
from a1 and a2 are byte-identical:
36a87b5e5cb0e74b3eed2b73d3c63f90ad6e91a3cf690686fa56d8559bf38fb1.

All 143 pre-existing host tests pass. Host-test and native-test logs are archived.
The observed estimator maximum fuel/frame remains the compute criterion; neither
the whole-suite budget nor the generous sandbox execution ceiling is a claimed
performance result.

The new reproduction wrapper has command-line and syntax checks. Its full replay
was not separately run during this experiment; native estimator repetition on
every evaluated flight is checked by the primary evaluator. A reproduction must
use an unused structural-lines-v13-repro-* namespace and already-complete data,
and should run without concurrent generation/build/training for wall timing.

Windows files use new V13 names and are verified against the evaluated hashes.
They are experimental diagnostics, not replacements for a formal submission.
