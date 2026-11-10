"""Reproduce the illustrative figure from the frozen auxiliary validation cache."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CACHE = ROOT / "data/rotation-aux-v3"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


record = json.loads((CACHE / "cache.json").read_text())
assert record["status"] == "complete"
assert digest(CACHE / "validation.npz") == record["splits"]["validation"]["sha256"]
data = np.load(CACHE / "validation.npz", allow_pickle=False)
examples = []
for axis_id in range(3):
    candidates = np.flatnonzero((data["axis_id"] == axis_id) &
                               (data["angle_degrees"] >= 1.5) & (data["angle_degrees"] <= 2.5))
    # Texture makes subpixel perspective changes legible; this is illustration,
    # independent of any model's predictions or performance.
    contrast = data["images"][candidates, 0].std(axis=(1, 2))
    index = int(candidates[np.argmax(contrast)])
    episode = record["splits"]["validation"]["episodes"][int(data["episode_id"][index])]
    examples.append({"cache_index": index, "axis_id": axis_id,
                     "signed_angle_degrees": float(data["signed_angle_degrees"][index]),
                     "rotation_vector_gt_radians": data["rotation_vector_gt"][index].tolist(),
                     "episode_id": int(data["episode_id"][index]),
                     "source_episode": episode["path"], "scene": episode["scene"],
                     "source_frame_index": int(data["frame_index"][index]),
                     "minimum_source_clearance_pixels": float(data["source_clearance_pixels"][index]),
                     "previous_intensity_std": float(contrast.max())})

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(3, 3, figsize=(8.5, 6.1))
fig.subplots_adjust(left=.16, right=.90, top=.84, bottom=.065, wspace=.055, hspace=.13)
axis_names = [("Camera x", "pitch / right axis"), ("Camera y", "yaw / down axis"), ("Camera z", "roll / forward axis")]
for row, example in enumerate(examples):
    pair = data["images"][example["cache_index"]]
    axes[row, 0].imshow(pair[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[row, 1].imshow(pair[1], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    diff = axes[row, 2].imshow(np.abs(pair[1] - pair[0]), cmap="magma", vmin=0, vmax=.25, interpolation="nearest")
    name, description = axis_names[row]
    axes[row, 0].set_ylabel(f"{name}\n{description}\n{example['signed_angle_degrees']:+.4f}°",
                            rotation=0, ha="right", va="center", labelpad=8)
    for axis in axes[row]:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(.5)
            spine.set_color("#888888")
for column, title in enumerate(("Previous view", "Current view", "Absolute difference")):
    axes[0, column].set_title(title, pad=8, fontsize=11)
bar = fig.colorbar(diff, cax=fig.add_axes([.925, .18, .014, .48]), ticks=[0, .125, .25])
bar.set_label("Grayscale intensity difference", fontsize=9)
fig.suptitle("Perspective changes under pure camera-centre rotation", fontsize=14, y=.975)
fig.text(.5, .915, "Held-out synthetic validation examples · actual 64 × 48 model inputs", ha="center", fontsize=10)
fig.savefig(OUT / "rotation-examples.png", dpi=180, facecolor="white", bbox_inches="tight", pad_inches=.15)
plt.close(fig)
evidence = {"figure": "rotation-examples.png", "cache_json_sha256": digest(CACHE / "cache.json"),
            "validation_cache_sha256": digest(CACHE / "validation.npz"),
            "selection": "For each x/y/z axis, highest previous-image intensity standard deviation among validation examples with absolute angle 1.5 to 2.5 degrees. No model predictions used.",
            "display": "Nearest-neighbour display enlargement; input grayscale [0,1]; absolute difference magma scale [0,0.25], values above0.25 saturated.",
            "examples": examples}
(OUT / "rotation-examples.json").write_text(json.dumps(evidence, indent=2) + "\n")
caption = """Pure camera-centre rotation examples from the frozen auxiliary validation split (ProcTHOR scenes 5/6). Each row shows the previous and current 64×48 grayscale network inputs and their absolute pixel difference. The two views use symmetric ±half-angle SO(3) perspective reprojection of the same source observation, with a fixed crop and updated intrinsics; all sampled rays remain inside the source image. This is geometric image synthesis, not a newly rendered robot motion or a learned image generator.

Labels give the signed relative rotation around the OpenCV camera axes: x points right, y down and z forward. The right column uses a fixed [0,0.25] grayscale-difference colour scale, saturating larger differences; input images use [0,1]. Display enlargement uses nearest-neighbour interpolation. For each axis, the highest-contrast source among validation samples with 1.5–2.5° rotation was chosen to make the small changes visible; no model prediction or performance determined selection. Exact sample indices, source frames, angles and cache hashes are in `rotation-examples.json`. These examples illustrate the training signal and do not demonstrate estimation accuracy.
"""
(OUT / "rotation-examples-caption.md").write_text(caption, encoding="utf-8")
print(json.dumps({"figure": str(OUT / "rotation-examples.png"), "examples": examples}, indent=2))
