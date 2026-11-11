"""Render the archived v5 comparison, with aggregation and selection visible."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/feature-correlation-v5"


def main():
    rows = json.loads((OUT / "comparison.json").read_text())[:5]
    labels = ["Baseline\nKLT + IMU", "Previous\nCNN", "Calibrated\ngyro (v4)", "Single-frame\ncontrol", "Temporal\ncorrelation"]
    colors = ["#87949e", "#87949e", "#247aac", "#87949e", "#e3843b"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), constrained_layout=False)
    for ax, key, title, unit, digits in zip(axes,
        ["mean_sim3_ate_m", "median_rotation_deg", "median_step_ms"],
        ["Aligned trajectory error", "Rotation error", "WASM step time"],
        ["Mean Sim(3) ATE (m)", "Median episode RMSE (deg)", "Median step (ms)"], [3, 2, 2]):
        values = [r[key] for r in rows]
        bars = ax.bar(range(5), values, color=colors, width=.7)
        ax.set_xticks(range(5), labels, fontsize=8)
        ax.set_ylabel(unit, fontsize=10); ax.set_title(title, fontsize=12)
        ax.set_ylim(0, max(values)*1.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.16); ax.set_axisbelow(True)
        ax.bar_label(bars, labels=[f"{v:.{digits}f}" for v in values], padding=4, fontsize=9)
    fig.suptitle("AeroVIO: lightweight feature-correlation experiment", fontsize=16, y=.98)
    fig.text(.5, .89, "10 public simulated flights per variant · lower is better · same runtime and limits", ha="center", fontsize=10)
    fig.text(.5, .035, "Both new branches failed the earlier validation rule. Preserved v4 reference remains; public scores do not select the model.\nSim(3) removes scale. Step time is local execution time. One training seed; no physical-robot result.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.78, bottom=.22, left=.06, right=.99, wspace=.31)
    path = OUT / "comparison.png"
    fig.savefig(path, dpi=180)
    def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / "plot-provenance.json").write_text(json.dumps({"source_sha256": digest(Path(__file__)),
        "comparison_sha256": digest(OUT / "comparison.json"), "figure_sha256": digest(path),
        "selection": "first five archived rows; deterministic temporal repeat excluded from bars", "zero_based_axes": True}, indent=2) + "\n")


if __name__ == "__main__": main()
