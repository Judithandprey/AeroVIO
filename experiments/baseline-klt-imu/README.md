# Official KLT + IMU baseline

**Verified locally on November 10, 2026:** unchanged upstream example, WASM build,
20-frame synthetic check, and all ten public episodes. See `source.json`,
`build.json`, `verification.json`, and [upstream attribution](../../THIRD_PARTY_NOTICES.md).

## Results

| Measure | Observed result |
| --- | --- |
| Completed episodes | 10/10 |
| Frames attempted | 12,000 |
| Failed/filled frames | 0 |
| Equal-episode mean Sim(3) ATE | 2.041634864 m |
| Maximum fuel per frame | 141,910,846 |
| Mean fuel per frame | 60,274,322.674 |
| Median unaligned ATE across episodes | 41.428754 m |
| Median frame wall time | 3.281 ms, host-specific |
| Solution WASM size | 50,029,824 bytes |

Wall time is observational and is not the deterministic compute score. The
synthetic check has different image content and correspondingly different fuel;
it is not included in the public score.

| Episode | Hash prefix | Sim(3) ATE, m | Unaligned ATE, m | Max fuel/frame | Filled |
| --- | --- | ---: | ---: | ---: | ---: |
| 00 | `0386e5c9` | 2.099235 | 450.613 | 96,767,831 | 0 |
| 01 | `9b6a9279` | 1.914120 | 692.833 | 128,921,610 | 0 |
| 02 | `6e8e3766` | 2.499868 | 27.495 | 112,516,201 | 0 |
| 03 | `f8fb89ff` | 1.928537 | 37.573 | 110,577,010 | 0 |
| 04 | `489ad518` | 1.074172 | 149.903 | 92,238,317 | 0 |
| 05 | `51ea6311` | 2.330615 | 20.802 | 128,219,402 | 0 |
| 06 | `60ed4323` | 2.197604 | 47.523 | 141,910,846 | 0 |
| 07 | `8fff5c8d` | 2.156139 | 18.047 | 107,702,849 | 0 |
| 08 | `41e11443` | 2.619881 | 36.894 | 109,326,087 | 0 |
| 09 | `8321da1f` | 1.596178 | 45.285 | 96,744,104 | 0 |

The baseline executes successfully but has substantial drift and scale error on
several trajectories. Sim(3) alignment removes a global scale, rotation and
translation before measuring positional error; it does not establish a metric
3D map. For example, episode 00 uses an alignment scale of
0.012924928. Preserve these weaknesses as
baseline observations instead of changing the reference implementation.

## Repeatability and source verification

The first public episode was rerun using the same binary, inputs, runner and
limits. All trajectory metrics, all 1,200 per-frame fuel values, `estimate.tum`
and `twist.txt` matched exactly. Wall time differed as expected. The other nine
episodes were not repeated.

The source extracted from the built WASM, tracked `solution.py`, and the pinned
upstream example are byte-identical. Runtime and solution hashes are recorded.

## Evidence and reproduction

- `baseline-public/report.json`: complete upstream report for all ten episodes.
- `baseline-public/manifest.json`: commands, versions, selection, hashes, exit code.
- `baseline-repeat/`: repeat report and provenance for episode 00.
- `check.json` and `smoke-check.txt`: 20-frame ABI smoke evidence.
- `../../configs/`: public episode manifest, limits, and host dependency snapshot.
- [Setup commands](../../docs/setup.md).

Raw trajectories, fuel arrays, full build logs, binaries and cached datasets
remain local under ignored output/cache directories. Compact reports are tracked.
The same selection and limits are required for subsequent comparisons.

This is local public-set evidence. No course-server submission or held-out
assessment was performed, and saved limits are not confirmed server settings.
