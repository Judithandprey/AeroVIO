# Rotation-focused training audit

Both predeclared selection runs retained **epoch 0**, the original image+IMU warping checkpoint. Its ONNX hash is `0271431fc621b160cabad4b448cedc91f9ea4e355f080a366823273041afdbb7`. The trained final candidates are retained for diagnosis and are explicitly marked **unselected**. No public scores entered training or checkpoint selection.

## Selection outcome

All values below come from the four original scene-disjoint real validation episodes or the 800 separate synthetic rotation validation pairs. Real rotation is the arithmetic mean of per-episode accumulated orientation RMSE, including the initial zero-error frame; positions are unaligned.

Here `real_validation` is the code's name for original **simulated** flights with
translation and rotation, contrasted with pure-rotation augmentation. It does
not refer to physical-robot measurements.

| Candidate | Epoch | Real rotation RMSE (deg) | Pair translation RMSE (m) | Trajectory position RMSE (m) | Synthetic rotation RMSE (deg) | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Starting model | 0 | 6.948942 | 0.01103480 | 0.51870046 | 0.487619 | Retained |
| control: best trained rotation | 17 | 6.940446 | 0.01119482 | 0.55882160 | 0.566637 | trajectory position guard failed |
| control: fixed final diagnostic | 20 | 6.954437 | 0.01120302 | 0.55968595 | 0.567904 | rotation did not improve; trajectory position guard failed |
| focus: best trained rotation | 1 | 7.046742 | 0.01115806 | 0.51329639 | 0.438551 | rotation did not improve |
| focus: fixed final diagnostic | 20 | 7.699465 | 0.01124314 | 0.55214851 | 0.237649 | rotation did not improve; trajectory position guard failed |

The fixed 1.05x guards are pair translation RMSE <= **0.01158654 m** and mean trajectory position RMSE <= **0.54463548 m**. Both must pass before any lower real rotation error can select a checkpoint. Ties retain the earlier checkpoint. Auxiliary validation never selects a checkpoint.

The continuation control slightly lowers real rotation error at epoch 17, but its accumulated position error exceeds the guard. The focused run improves the synthetic rotation task while worsening real accumulated rotation: its final auxiliary error drops from 0.487619 to 0.237649 degrees, while real rotation rises from 6.948942 to 7.699465 degrees. This shows a transfer failure under this fixed protocol; it does not establish that visual features, rather than the IMU/state branch, caused the synthetic gain.

## Matched training and extra compute

Both variants start from the same checkpoint tensors, use its unchanged input/target normalization, train all 29,398 parameters, process the same 47,760 real examples in identical minibatch order, and perform exactly 760 Adam steps. Each run uses 20 epochs, seed 17, batch 64, fresh Adam, and the same cosine learning-rate schedule. The focused variant additionally processes 24,320 auxiliary examples in batches of 32 using an independent sampler seed. Only its auxiliary SO(3) gradient is added before the shared clipping/optimizer step; auxiliary translation labels are unused.

| Variant | Initial run total (s) | Retention repeat total (s) | Initial extra auxiliary forward/backward (s) | Retention repeat extra auxiliary forward/backward (s) |
| --- | ---: | ---: | ---: | ---: |
| control | 6.805338 | 6.252619 | 0.000000 | 0.000000 |
| focus | 8.700327 | 9.328880 | 2.399832 | 2.475543 |

Total elapsed time covers the epoch loop and export, excluding initial loading/audits. Auxiliary timing includes sampling, forward evaluation, loss and backward propagation; it excludes the shared gradient clipping, optimizer and validation. These local training timings are separate from deployed WASM latency/fuel. The final-candidate export makes the retention repeat do additional post-training work.

## Retention-only repeat and provenance

The first trainer retained the selected checkpoint only. After both selected epoch 0, its exact source bytes were preserved at [train_rotation_focus.initial.py](provenance/train_rotation_focus.initial.py). The only subsequent change saves `last.pt` and exports the fixed final epoch after optimization and selection finish. Objectives, initialization, normalization, minibatches, optimizer, schedule and checkpoint rules remain unchanged. Both runs were repeated into new directories.

Every history field except elapsed time repeats exactly for all 21 entries (epoch 0 through 20), including every accuracy/loss, eligibility decision and selected flag. Selected ONNX and metadata are byte-identical to their initial-run counterparts. Source-hash differences are confined to the retention-only trainer edit. See [the change record](provenance/retention-only-change.json) and [machine-readable audit](training-audit.json).

The host validation integrator was independently exercised through the actual `Solution.step` composition code on all four validation episodes (796 pairs). Maximum rotation/position RMSE differences were 2.22e-16 rad / 5.55e-16 m; the starting PyTorch checkpoint and ONNX agree within 4.77e-7 in normalized output. This verifies host composition and export contracts; it is not a full WASM evaluation. See [rollout audit](rollout-audit.json).

## Archived records

- `control/` and `focus/`: retained selected ONNX, metadata, run manifest and full history.
- Each `initial-run/`: original manifest and history, preserved without rewriting their source hashes.
- Each `diagnostic-last/`: fixed epoch-20 diagnostic ONNX and metadata (`selected_for_deployment=false`).
- Local `.pt` files and training logs remain in ignored `models/` and `logs/`; log paths and hashes are recorded in `training-audit.json`.

All deployment architecture, image resolution, state normalization and runtime rotation behavior remain unchanged. The public comparison, if run, must distinguish selected epoch-0 models from the fixed unselected final diagnostics and must not feed back into selection.
