"""Plot x/y/z position over time: ground truth vs one or more Sim(3)-aligned estimates.

Three stacked panels (x, y, z in meters) share a time axis. Ground truth is green;
each estimate is Sim(3)-aligned (scale + rotation + translation, Umeyama's closed-form
method, solved on positions only) to that same ground truth and drawn in its own color.

Two modes:
  - One report directory: plot a single build. --episode picks the flight (hash
    prefix); omit it to auto-pick the flight closest to the report's median Sim(3)
    ATE. --all-episodes writes one PNG per completed episode instead.
  - Several report directories (repeat the positional argument): overlay every
    build's estimate for the *same* episode, so you can compare builds directly.
    All report directories must share that episode (e.g. all evaluated against
    viobench's public set). Label each with --labels, in the same order; default
    labels are the report directory names.

Ground truth comes from the raw episode archive, found by content hash:
  - viobench's official public/held-out sets resolve through conta automatically
    (network access, no local data needed) -- just point --data at an empty or
    missing directory, or omit it; conta's own cache (~/.cache/conta) is checked
    first regardless.
  - Custom-generated sets (e.g. val-agile/val-fast/val-narrow/val-nohover) are not
    in conta's index. Raw episodes for those are gitignored (see README); run this
    on a machine that actually has that data/ directory, and pass it via --data.

Usage:
    # single build, auto-picked representative episode
    .venv/bin/python scripts/plot_trajectory.py results-for-tae/eval/pub

    # single build, every public episode
    .venv/bin/python scripts/plot_trajectory.py results-for-tae/eval/pub --all-episodes

    # compare builds on one fixed episode
    .venv/bin/python scripts/plot_trajectory.py eval-v8-rotfix eval-v8-scale2 results-for-tae/eval/pub \
        --labels "V8 rotfix (scale1.0)" "V8 scale2.0" "window h128" \
        --episode 41e11443a9a5cc165ef9aeeaef11dba63614685b

    # a custom (non-public) set, data available locally
    .venv/bin/python scripts/plot_trajectory.py results/val-agile/report-tae-v8 --data data/val-agile
"""
import argparse
import hashlib
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viobench.tools.datasets.storage import read_episode

COLORS = ("#E07B1C", "#2465A8", "#8E44AD", "#C0392B", "#16A085")
GT_COLOR = "#2E8B3D"


def index_archives(directories):
    """Same convention as scripts/summarize_eval.py: key archives by their SHA-1."""
    found = {}
    for directory in directories:
        directory = Path(directory).expanduser()
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.suffix == ".zip":
                found[hashlib.sha1(path.read_bytes()).hexdigest()] = path
            elif len(path.name) == 40 and path.is_file():
                found[path.name] = path
    return found


def ground_truth_path(episode_hash, data_dirs):
    """A path viobench can read for this episode: a locally indexed archive if one
    matches, otherwise a bare conta reference (conta resolves/downloads it)."""
    archives = index_archives(data_dirs)
    return archives.get(episode_hash, f"conta:{episode_hash}")


def umeyama(src, dst):
    """Least-squares similarity transform (scale, rotation, translation) mapping
    src points onto dst points (Umeyama 1991). Returns (s, R, t) with
    dst ~= s * (R @ src.T).T + t."""
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = (dst_c.T @ src_c) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_src = (src_c ** 2).sum(axis=1).mean()
    s = np.trace(np.diag(D) @ S) / var_src
    t = mu_dst - s * R @ mu_src
    return s, R, t


def load_report(report_dir):
    return json.loads((Path(report_dir) / "report.json").read_text())


def pick_episode(report, episode_hash):
    episodes = report["episodes"]
    if episode_hash is not None:
        matches = [e for e in episodes if e["hash"].startswith(episode_hash)]
        if len(matches) != 1:
            raise ValueError(f"{len(matches)} episodes match prefix {episode_hash!r}")
        return matches[0]
    # Default: the completed episode whose Sim(3) ATE is closest to the report median,
    # so the plot is representative rather than the best- or worst-case flight.
    completed = [e for e in episodes if e["completed"]]
    target = float(np.median([e["metrics"]["sim3_ate_rmse"] for e in completed]))
    return min(completed, key=lambda e: abs(e["metrics"]["sim3_ate_rmse"] - target))


def load_estimate(report_dir, episode_hash):
    rows = np.loadtxt(Path(report_dir) / "logs" / episode_hash / "estimate.tum", comments="#")
    return rows[:, 0], rows[:, 1:4]


def load_ground_truth(episode_hash, data_dirs):
    episode = read_episode(ground_truth_path(episode_hash, data_dirs), frames="none")
    return episode.ground_truth_relative()[:, :3]


def plot_single(report_dir, episode_hash, data_dirs, output):
    report = load_report(report_dir)
    episode = pick_episode(report, episode_hash)
    episode_hash = episode["hash"]
    t, est = load_estimate(report_dir, episode_hash)
    gt = load_ground_truth(episode_hash, data_dirs)
    if len(est) != len(gt):
        raise ValueError(f"{episode_hash}: {len(est)} estimated poses vs {len(gt)} ground truth")
    scale, rotation, translation = umeyama(est, gt)
    aligned = scale * (rotation @ est.T).T + translation

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 7.4), sharex=True)
    for ax, series_gt, series_est, label in zip(axes, gt.T, aligned.T, "xyz"):
        ax.plot(t, series_gt, color=GT_COLOR, linewidth=1.4, label="ground truth")
        ax.plot(t, series_est, color=COLORS[0], linewidth=1.2, label="estimate (Sim(3))")
        ax.set_ylabel(f"{label} [m]")
        ax.grid(alpha=0.15)
    axes[0].legend(loc="best", fontsize=9, frameon=False)
    axes[-1].set_xlabel("t [s]")
    sim3_ate = episode["metrics"]["sim3_ate_rmse"]
    fig.suptitle(f"{Path(report_dir).name} / {episode_hash[:8]} \N{MIDDLE DOT} "
                 f"Sim(3) ATE {sim3_ate:.3f} m \N{MIDDLE DOT} scale {scale:.2f}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"wrote {output} (episode {episode_hash}, scale={scale:.3f})")


def plot_batch(report_dir, data_dirs, out_dir):
    report = load_report(report_dir)
    out_dir = Path(out_dir) if out_dir else Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for episode in report["episodes"]:
        if not episode["completed"]:
            print(f"skip {episode['hash']}: not completed")
            continue
        output = out_dir / f"trajectory-{episode['hash'][:8]}.png"
        plot_single(report_dir, episode["hash"], data_dirs, output)


def plot_compare(report_dirs, labels, episode_hash, data_dirs, output):
    reports = [load_report(d) for d in report_dirs]
    if episode_hash is None:
        # Default: the first report's representative episode, so every run is
        # directly comparable (and reproducible run to run).
        episode_hash = pick_episode(reports[0], None)["hash"]
    t = None
    aligned_by_label = []
    for report_dir, report, label in zip(report_dirs, reports, labels):
        matches = [e for e in report["episodes"] if e["hash"] == episode_hash]
        if not matches:
            raise ValueError(f"{report_dir}: no episode {episode_hash}")
        episode = matches[0]
        t_i, est = load_estimate(report_dir, episode_hash)
        gt = load_ground_truth(episode_hash, data_dirs)
        if len(est) != len(gt):
            raise ValueError(f"{report_dir}/{episode_hash}: {len(est)} estimated poses vs {len(gt)} ground truth")
        scale, rotation, translation = umeyama(est, gt)
        aligned = scale * (rotation @ est.T).T + translation
        aligned_by_label.append((label, aligned, episode["metrics"]["sim3_ate_rmse"], scale))
        t = t_i

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 8.2), sharex=True, layout="constrained")
    for ax, axis_index, axis_label in zip(axes, range(3), "xyz"):
        ax.plot(t, gt[:, axis_index], color=GT_COLOR, linewidth=1.6, label="ground truth", zorder=10)
        for (label, aligned, ate, scale), color in zip(aligned_by_label, COLORS):
            ax.plot(t, aligned[:, axis_index], color=color, linewidth=1.1, alpha=0.9,
                     label=f"{label} \N{MIDDLE DOT} ATE {ate:.3f} m (scale {scale:.2f})")
        ax.set_ylabel(f"{axis_label} [m]")
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("t [s]")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", fontsize=8.5, frameon=False,
               ncol=1, title=f"episode {episode_hash[:8]}", title_fontsize=11)
    output = output or Path(report_dirs[0]) / f"compare-{episode_hash[:8]}.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"wrote {output} (episode {episode_hash})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report_dir", type=Path, nargs="+",
                         help="one directory: plot that build; several: overlay them on one shared episode")
    parser.add_argument("--data", nargs="+", default=[],
                         help="directories to search for a raw episode archive by SHA-1 filename, "
                              "before falling back to conta (network download of the public/held-out sets)")
    parser.add_argument("--episode", help="episode hash (prefix ok); default: closest-to-median Sim(3) ATE")
    parser.add_argument("--all-episodes", action="store_true",
                         help="single-report mode only: write one PNG per completed episode")
    parser.add_argument("--out-dir", type=Path, help="--all-episodes only: where to write the PNGs")
    parser.add_argument("--labels", nargs="+", help="multi-report mode only: one label per report_dir, in order")
    parser.add_argument("--output", type=Path, help="output PNG path (single-episode modes)")
    args = parser.parse_args()

    if len(args.report_dir) == 1:
        report_dir = args.report_dir[0]
        if args.all_episodes:
            if args.output:
                parser.error("--output is not used with --all-episodes; use --out-dir")
            plot_batch(report_dir, args.data, args.out_dir)
        else:
            output = args.output or report_dir / f"trajectory-{(args.episode or 'auto')[:8]}.png"
            plot_single(report_dir, args.episode, args.data, output)
    else:
        if args.all_episodes:
            parser.error("--all-episodes needs exactly one report_dir")
        labels = args.labels or [d.name for d in args.report_dir]
        if len(labels) != len(args.report_dir):
            parser.error(f"got {len(args.report_dir)} report dirs but {len(labels)} labels")
        plot_compare(args.report_dir, labels, args.episode, args.data, args.output)


if __name__ == "__main__":
    main()
