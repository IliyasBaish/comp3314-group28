"""Plot only recorded results; never manufacture a pending LGC comparison."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/baseline"))
    args = parser.parse_args(argv)
    root = args.results
    manifest = json.loads((root / "manifest.json").read_text())
    seed = manifest["seeds"][0]  # Fixed by the protocol; never choose the prettiest seed.
    points = [row for row in read_csv(root / "points.csv") if int(row["seed"]) == seed]
    predictions = [row for row in read_csv(root / "predictions.csv") if int(row["seed"]) == seed]
    X = np.array([[float(row["x0"]), float(row["x1"])] for row in points])
    observed = np.array([int(row["observed_label"]) for row in points])
    groups = {}
    for row in predictions:
        groups.setdefault((row["method"], row["sigma"]), []).append(int(row["prediction"]))
    # First declared bandwidth for an illustration; D can use CSVs for a full sweep.
    shown = [(key, values) for key, values in groups.items()
             if key[0] == "1-NN" or float(key[1]) == manifest["sigmas"][0]]
    fig, axes = plt.subplots(1, 1 + len(shown), figsize=(5 * (1 + len(shown)), 4.8), sharex=True, sharey=True)
    axes[0].scatter(X[:, 0], X[:, 1], c="#bcc3cc", s=15)
    axes[0].set_title("Input: only two labels revealed")
    for ax, ((method, sigma), labels) in zip(axes[1:], shown):
        ax.scatter(X[:, 0], X[:, 1], c=np.array(["#2563eb", "#e87924"])[labels], s=15)
        ax.set_title(method + (f" predictions (sigma={sigma})" if sigma else " predictions"))
    for ax in axes:
        revealed = observed >= 0
        ax.scatter(X[revealed, 0], X[revealed, 1],
                   c=np.array(["#2563eb", "#e87924"])[observed[revealed]], marker="*",
                   s=250, edgecolors="black", linewidths=1, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlabel("Feature 1")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Feature 2")
    fig.suptitle(f"Our generated two-moons data | seed {seed} | stars = revealed labels", fontsize=13)
    fig.text(0.5, 0.025, "Partial qualitative reproduction. " +
             ("LGC comparison pending." if manifest["predictor"] is None else "First configured bandwidth shown; not selected by error."),
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(root / "example.png", dpi=180)
    plt.close(fig)

    summary = read_csv(root / "summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = [row["method"] + (f"\nsigma={row['sigma']}" if row["sigma"] else "") for row in summary]
    means = np.array([float(row["mean_error"]) * 100 for row in summary])
    deviations = np.array([float(row["std_error"]) * 100 for row in summary])
    ax.bar(labels, means, width=0.55, yerr=deviations, capsize=6, color=["#64748b" if row["method"] == "1-NN" else "#2563eb" for row in summary])
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.set_ylabel("Error on unrevealed points (%)")
    ax.set_ylim(0, min(100, max(30, max(means + deviations) + 10)))
    ax.set_title(f"Our toy experiment: mean +/- sample SD ({len(manifest['seeds'])} seeds)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.025, "Descriptive variation across datasets and label placements; not a confidence interval.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(root / "error_summary.png", dpi=180)
    plt.close(fig)
    print(f"Saved {root / 'example.png'} and {root / 'error_summary.png'}")


if __name__ == "__main__":
    main()
