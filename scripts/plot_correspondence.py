"""Standalone v6 comparison figure, including the direct matching diagnostic."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/correspondence-v6"


def main():
    rows = json.loads((OUT / "comparison.json").read_text())[:6]
    matching = json.loads((OUT / "matching-audit.json").read_text())["matching"]["validation"]
    labels = ["Baseline", "v4 gyro", "v5 single", "v5 corr.", "v6 cont.", "v6 labels"]
    colors = ["#8c99a3", "#8c99a3", "#8c99a3", "#247aac", "#8c99a3", "#e3843b"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specifications = [(axes[0, 0], "mean_sim3_ate_m", "Public trajectory error", "Mean Sim(3) ATE (m)", 3),
                      (axes[0, 1], "median_rotation_deg", "Public rotation error", "Median episode RMSE (deg)", 2),
                      (axes[1, 0], "median_step_ms", "Public WASM step time", "Median step (ms)", 2)]
    for ax, key, title, ylabel, digits in specifications:
        values = [r[key] for r in rows]
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, labels=[f"{v:.{digits}f}" for v in values], padding=3, fontsize=9)
        ax.set_ylim(0, max(values)*1.2); ax.set_title(title); ax.set_ylabel(ylabel)
    ax = axes[1, 1]
    values = [matching["v5-temporal"]["all"]["mean_episode_zero_epe_pixels"],
        *[matching[n]["all"]["mean_episode_epe_pixels"] for n in ("v5-temporal", "continuation", "supervised")]]
    bars = ax.bar(["Zero motion", "v5 corr.", "v6 cont.", "v6 labels"], values,
                  color=["#8c99a3", "#247aac", "#8c99a3", "#e3843b"])
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3, fontsize=9)
    ax.set_ylim(0, max(values)*1.2); ax.set_title("Validation correspondence error")
    ax.set_ylabel("Mean episode EPE (64x48 image pixels)")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True); ax.tick_params(axis="x", labelsize=9)
    fig.suptitle("AeroVIO: explicit correspondence-supervision experiment", fontsize=16, y=.98)
    fig.text(.5, .025, "Lower is better. The supervised candidate failed pre-public validation; previous versions remain unchanged.\n10 public flights per run; 4 scene-disjoint validation flights. One seed, local WASM timing, no robot result.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.89, bottom=.14, left=.08, right=.98, hspace=.4, wspace=.25)
    path = OUT / "comparison.png"; fig.savefig(path, dpi=180)
    def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / "plot-provenance.json").write_text(json.dumps({"script_sha256": digest(Path(__file__)),
        "comparison_sha256": digest(OUT / "comparison.json"), "matching_audit_sha256": digest(OUT / "matching-audit.json"),
        "figure_sha256": digest(path), "zero_axes": True, "matplotlib": matplotlib.__version__}, indent=2)+"\n")


if __name__ == "__main__": main()
