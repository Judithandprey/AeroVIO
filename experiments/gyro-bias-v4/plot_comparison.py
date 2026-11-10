"""Plot the five frozen v4 public summaries without running inference.

Requires matplotlib and numpy. Run from any working directory; default inputs
and outputs are alongside this script. Earlier plot evidence is not overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORDER = ("baseline", "current", "gyro-only", "calibrated-residual", "calibrated-gyro")
LABELS = (
    "Official KLT + IMU\nbaseline",
    "Current + residual\nselected fallback",
    "Gyro-only\nresidual disabled",
    "Calibrated + residual\nunselected",
    "Calibrated gyro-only\ndiagnostic, unselected",
)
COLORS = ("#808C99", "#2465A8", "#B5C3D1", "#47A6A0", "#D99532")
METRICS = (
    ("sim3_ate_mean_m", "Mean Sim(3) ATE", "meters", 1.0),
    ("rotation_rmse_median_deg", "Median rotation RMSE", "degrees", 1.0),
    ("fuel_per_frame_max", "Maximum fuel / frame", "million fuel units", 1e-6),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=HERE / "comparison.json")
    parser.add_argument("--output", type=Path, default=HERE / "comparison.png")
    parser.add_argument("--provenance", type=Path, default=HERE / "comparison-plot-provenance.json")
    args = parser.parse_args()
    if args.output.exists() or args.provenance.exists():
        parser.error("preserve earlier figure/provenance; use new output paths")
    source_hash = sha(args.comparison)
    records = json.loads(args.comparison.read_text())
    indexed = {record["name"]: record for record in records}
    if len(indexed) != len(records) or any(name not in indexed for name in ORDER):
        raise ValueError("comparison needs unique names and all five primary variants")
    rows = [indexed[name] for name in ORDER]
    if any(row["episodes"] != 10 or row["completed"] != 10 or row["filled_frames"] != 0 for row in rows):
        raise ValueError("figure requires ten complete, unfilled episodes per primary variant")
    selection_path = HERE / "selection.json"
    selection = json.loads(selection_path.read_text())
    if selection["selected_variant"] != "current" or not selection["validation_complete"]:
        raise ValueError("figure roles require the recorded retained-current decision")
    values = np.array([[float(row[key]) * scale for row in rows] for key, _, _, scale in METRICS])
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("invalid plotted metric")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelcolor": "#334155", "text.color": "#172A3A",
                         "xtick.color": "#475569", "ytick.color": "#172A3A"})
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 6.8), sharey=True)
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.24, top=0.78, wspace=0.22)
    y = np.arange(len(rows))
    axis_limits = []
    for ax, metric, plotted in zip(axes, METRICS, values):
        _, title, unit, _ = metric
        bars = ax.barh(y, plotted, height=0.60, color=COLORS, edgecolor="white", linewidth=1.0)
        bars[-1].set_hatch("//")
        bars[-1].set_edgecolor("#7B4A10")
        bars[-1].set_linewidth(0.6)
        maximum = float(np.max(plotted))
        ax.set_xlim(0, maximum * 1.40)
        axis_limits.append(list(ax.get_xlim()))
        ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=16)
        ax.set_xlabel(unit, fontsize=11, labelpad=9)
        ax.xaxis.grid(True, color="#DFE5EC", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_yticks(y, LABELS)
        ax.tick_params(axis="y", length=0, pad=12, labelsize=11)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#B6C2CF")
        for bar, value in zip(bars, plotted):
            ax.text(value + maximum * 0.025, bar.get_y() + bar.get_height() / 2,
                    f"{value:.6f}", va="center", ha="left", fontsize=10,
                    color="#172A3A")
    axes[0].invert_yaxis()
    fig.suptitle("AeroVIO v4: public accuracy and compute", x=0.035, y=0.955,
                 ha="left", fontsize=22, fontweight="bold")
    fig.text(0.035, 0.885, "Five primary variants  |  Lower is better in each panel  |  All axes start at zero",
             fontsize=12, color="#475569")
    caption = (
        "Local public set: 10 simulated flights per variant, all completed. No confidence intervals or statistical-superiority claim.\n"
        "Selection was frozen on separate validation: current remains the fallback; calibrated gyro-only is an unselected diagnostic.\n"
        "ATE is the equal-episode mean; rotation is the episode median; fuel is the maximum over all frames. Labels show six decimals."
    )
    fig.text(0.035, 0.065, caption, ha="left", va="bottom", fontsize=10, linespacing=1.6, color="#475569")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor="white", metadata={"Software": "AeroVIO plot_comparison.py"})
    plt.close(fig)
    if sha(args.comparison) != source_hash:
        raise ValueError("comparison changed while rendering")
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Plot of frozen public summaries only; no inference, training, selection or confidence-interval estimation",
        "input": {"path": "experiments/gyro-bias-v4/comparison.json", "sha256": source_hash},
        "selection": {"path": "experiments/gyro-bias-v4/selection.json", "sha256": sha(selection_path),
                      "selected_variant": "current"},
        "public_episodes_sha256": sha(ROOT / "configs/public-episodes.json"),
        "script_sha256": sha(Path(__file__)),
        "figure": {"path": args.output.name, "sha256": sha(args.output), "pixels": [2688, 1088],
                   "axis_limits": axis_limits, "bar_values_unrounded": True, "label_decimal_places": 6},
        "metrics": [{"source_key": key, "title": title, "display_unit": unit, "scale": scale}
                    for key, title, unit, scale in METRICS],
        "variants": [{"name": row["name"], "display_label": label,
                      "wasm_sha256": row["wasm_sha256"],
                      "values": {key: row[key] for key, _, _, _ in METRICS}}
                     for row, label in zip(rows, LABELS)],
        "caption": caption,
        "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                        "numpy": np.__version__, "platform": platform.system()},
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"figure": str(args.output), "sha256": sha(args.output), "provenance": str(args.provenance)}))


if __name__ == "__main__":
    main()
