# Why the 24-line limit did not bound detection-frame fuel

This is read-only source analysis of the actual OpenCV 4.12.0 guest used by the
probe. No detector algorithm, threshold, runtime library or evaluation was
changed for this analysis.

Verified WSL source:
`/root/projects/AeroVIO/workspace/build/runtime/src/opencv/modules/imgproc/src/lsd.cpp`

SHA-256: `54e8d04a4da08d787c1beaba4cff6f3711a982746847f084fae3004d6184732d`

The probe measured maximum 16,048,279 fuel and mean 13,417,792.87 fuel per
invocation. Those totals include line detection, vanishing-direction fitting
and diagnostic serialization. They are not per-stage profiling measurements.
A fuel-exhaustion stack inside `ll_angle`/`stable_sort` identifies the instruction
executing when fuel ran out, not the percentage of total work attributable to
that stage.

## Work that happens before the line cap

In `lsd.cpp:446-530`, `flsd` runs the full detector before returning any lines:

1. Lines 456-465 blur the entire 160x120 input and resize it by the configured
   0.8 scale to 128x96. With sigma-scale 0.6, this uses sigma 0.75 and a 7x7
   Gaussian kernel (the code may implement the filter separably).
2. Lines 533-576 compute gradient magnitudes over the interior pixel grid and
   angles for gradients above the threshold.
3. Lines 579-590 push **every interior pixel**, including below-threshold ones,
   into `ordered_points`: exactly `(128-1)*(96-1) = 12,065` records.
4. Line 594 calls `std::stable_sort` on those records. The `n_bins=64` argument
   only quantizes the keys (lines 579-586); this implementation still uses a
   comparison sort rather than a fixed-bucket pass. The comparator at lines
   388-390 orders descending by that integer key.
5. Lines 482-530 scan ordered seeds, grow regions and fit rectangles. Selecting
   `LSD_REFINE_NONE` skips refinement and NFA checks at lines 498-509, but does
   not skip the preceding blur, gradient, sorting, growth or rectangle fitting.

Our `structural_lines.h:248` calls this whole LSD routine. Only after it returns
does `from_segments` filter and sort segments, stop at 24 retained lines
(`structural_lines.h:239`), and run bounded vanishing-direction extraction.
Thus the cap bounds downstream geometry, not the expensive upstream detector.

Calling the detector once every eight frames reduces invocation count and mean
cost. It does not reduce the work within a detection frame, so it cannot by
itself fix `max fuel/frame`. Merely moving the same call to another frame phase
can avoid coincident tracker work but cannot remove the detector's own peak.

## Proposed future compute-only experiment; not implemented

The smallest source-level candidate is a separate, explicitly versioned LSD
ordering experiment: replace comparison sorting with stable 64-bucket ordering,
preserving descending-key order and row-major ties, and omit NOTDEF seed
records that the subsequent loop always skips. This could remove sorting work
without intentionally changing eligible seed order. Exact output equivalence
and actual WASM fuel must be tested; no speedup amount or successful budget fit
is established by this suggestion. Do not modify the shared pinned OpenCV build
under the preserved V8 baseline.

A more substantial bounded alternative is a small edge-chain detector using
already available image derivatives: retain only a fixed number of strong
seeds/visited edge pixels, connect locally compatible gradients and fit straight
segments with explicit work limits. Reuse of existing gradients is possible in
principle, not free in the current interface:

- `workspace/visual_native/visual_direction.h:110-112` builds an optical-flow
  pyramid with derivatives enabled. OpenCV's
  `workspace/build/runtime/src/opencv/modules/video/src/lkpyramid.cpp:754-758`
  allocates alternating image/derivative entries; lines 815-829 compute the
  Scharr derivative into entry `level*2+1`.
- `workspace/visual_native/split_corners.h:77-93` also computes integer Sobel
  derivatives, but retains only products in a three-row ring; it does not leave
  a reusable full-frame gradient image.

The Scharr derivatives are not numerically the same gradient definition as
LSD's 2x2 operator, so an edge-chain replacement would require new detector and
geometry validation. If processing is split across frames, every work item must
retain its original captured image, intrinsics and pose timestamp. Mixing old
gradients with a newer frame's orientation would introduce a geometry bug.
Bounded scheduling can lower peak work by adding latency; that latency must be
reported and tested separately from precision.

Additional verified source hashes:

- `lkpyramid.cpp`: `fb4afb3e075d67d47010330581ddd4290a0121778780963ae297e40e22395f9d`
- `split_corners.h`: `3f24b3f002ae280fb719e04cdbeb0d18bfdeebe7de07ec499d108ae38d311a0a`
- `visual_direction.h`: `a45f7dc31e9e3a53ceeafccb8ae8891e8277dfe85af7cf415130b8cb28d5fce7`
