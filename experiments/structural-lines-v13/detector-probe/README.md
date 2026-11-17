# Structural-line detector feasibility probe

Completed November 16, 2026 (America/Los_Angeles), before pose-error evaluation.
This is negative feasibility evidence: the initial module is expensive and
usually lacks two independent vanishing directions at 160x120.

The actual guest uses OpenCV 4.12.0, built with the existing WASI toolchain
(Clang 23.1.0). Host OpenCV 5.0.0 is used only for contact-sheet drawing; all
reported line extraction, vanishing directions and fuel come from the WASM.
Guest SHA-256: `7288c39a8e2a680e8732d56d3456074317bb204b0273d2a2fd852db6677d78d8`.

## Fixed development-only protocol

The four original 200-frame validation flights are reused as development data.
Probe frames are exactly 0,8,...,192 (25 per flight, 100 total). Input is the
native tracker's integer-rounded 2x2 GREEN-channel average of adapted 320x240
RGB. Intrinsics use the matching half-pixel resize convention. No ground-truth
pose, depth, estimator error or future frame enters extraction or frame choice.

The detector keeps at most 24 lines of length >=24 pixels and at most three
vanishing directions, with at least three distinct supporting lines, <=0.8-degree
line-plane residual, middle-eigenvalue/trace >=0.004 and >=18-pixel center span.
It does not impose perpendicular room axes or known global directions.
No threshold changed after these development results were observed.

| Scene / seed | Mean retained lines | Frames with >=2 directions | Peak probe fuel |
| --- | ---: | ---: | ---: |
| Train-5 / 271840 | 3.64 | 0 / 25 | 12,639,762 |
| Train-5 / 271841 | 4.84 | 0 / 25 | 13,308,592 |
| Train-6 / 271842 | 5.28 | 0 / 25 | 16,048,279 |
| Train-6 / 271843 | 7.32 | 7 / 25 | 15,712,843 |

Across 100 frames, 67 have zero directions, 26 have one, and only seven have two
or more. All seven occur in seed 271843 at frames 0,8,16,24,32,40,176.
The mean probe fuel is 13,417,792.87 and maximum is 16,048,279. These measure
the isolated detector plus vanishing-direction extraction and diagnostic output;
they are not the full estimator's max fuel/frame. Lower invocation frequency
reduces average cost but does not remove a detection frame's peak cost.

All output doubles and per-probe fuel match an independent fresh-instance
repeat on every sampled frame. The native synthetic suite returns zero failed
checks (61,969,494 total test fuel). It covers finite/infinite vanishing points,
oblique families, blank/single-family images, degeneracy, duplicate fragments,
short clutter, caps and deterministic output. WASI stdout buffering retained
only the first test's printed line in `probe.log`; the zero return code is
checked before the data probe and recorded in `report.json`. No source or binary
was changed merely to reprint that log.

`contact-sheet.png` shows fixed frames 0,64,128,192 for each flight, plus first
zero-direction and first two-direction examples when those are different frames.
Green/orange/purple indicate supporting line groups; gray lines are retained but
not assigned to an accepted group. The selection uses detector status only.
Many room images contain visible edges but too few separate long supports per
direction. A single family is not independent yaw evidence.

## Reproduction and provenance

The initial source snapshot is under `source/`, with hashes in `report.json`.
The original WASM and raw build directory remain in WSL:
`results/structural-lines-v13/detector-probe/build/structural_line_probe.wasm`.
The complete original output remains in
`results/structural-lines-v13/detector-probe/development-initial/`.
Raw frames were not copied into this compact archive.

From a matching clean checkout, choose fresh build/output names:

```bash
cmake -S tests/line_probe -B results/line-probe-repro/build \
  -DCMAKE_TOOLCHAIN_FILE=/root/projects/AeroVIO/workspace/build/toolchain/wasi-sdk/share/cmake/wasi-sdk-p1.cmake \
  -DVIOBENCH_NATIVE_DIR=/root/projects/AeroVIO/workspace/runtime/native \
  -DVIOBENCH_OPENCV_DIR=/root/projects/AeroVIO/workspace/build/runtime/opencv-build
cmake --build results/line-probe-repro/build -j 2
.venv/bin/python scripts/probe_structural_lines.py \
  --wasm results/line-probe-repro/build/structural_line_probe.wasm \
  --out results/line-probe-repro/development
```

Wall time is excluded because independent simulation generation overlapped
the probe. Fuel remains deterministic. No pose-accuracy improvement is claimed
by this detector-only test; the separate three-arm estimator comparison is
still needed to establish correction behavior and full-frame peak fuel.
