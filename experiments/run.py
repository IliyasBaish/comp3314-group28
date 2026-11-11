"""Run with: python -m experiments.run (from the repository root)."""

import argparse
import csv
import importlib
import json
from pathlib import Path
import platform

import numpy as np

from .protocol import evaluate, make_trial, nearest_neighbor, predict_lgc


def load_predictor(spec: str):
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError("--predictor must be module:function, e.g. src.lgc:predict")
    predictor = getattr(importlib.import_module(module), name)
    if not callable(predictor):
        raise ValueError("--predictor must identify a callable.")
    return predictor


def summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["method"], row["sigma"], row["alpha"])
        groups.setdefault(key, []).append(row["error_rate"])
    return [
        {
            "method": method, "sigma": sigma, "alpha": alpha,
            "n_trials": len(values), "mean_error": float(np.mean(values)),
            "std_error": float(np.std(values, ddof=1)),
            "mean_accuracy": float(1.0 - np.mean(values)),
        }
        for (method, sigma, alpha), values in groups.items()
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.15])
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--predictor", help="Agreed core adapter module:function; omitted = 1-NN only")
    parser.add_argument("--out", type=Path, default=Path("results/baseline"))
    args = parser.parse_args(argv)
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Provide at least two distinct nonnegative seeds for a sample SD.")
    if not np.isfinite(args.alpha) or not 0 < args.alpha < 1:
        parser.error("alpha must be in (0, 1).")
    if (len(set(args.sigmas)) != len(args.sigmas)
            or any(not np.isfinite(s) or s <= 0 for s in args.sigmas)):
        parser.error("sigmas must be distinct, finite, and positive.")
    if args.predictor is None and args.sigmas != [0.15]:
        parser.error("Bandwidth sweeps require --predictor; 1-NN has no bandwidth.")
    predictor = load_predictor(args.predictor) if args.predictor else None
    rows = []
    paired = []
    trials = []
    predictions = []
    for seed in args.seeds:
        trial = make_trial(seed, args.n_samples, args.noise)
        trials.append(trial)
        baseline = nearest_neighbor(trial.X, trial.observed)
        baseline_metrics = evaluate(trial, baseline)
        rows.append({"seed": seed, "method": "1-NN", "sigma": "", "alpha": "", **baseline_metrics})
        predictions.append((seed, "1-NN", "", baseline))
        if predictor:
            for sigma in args.sigmas:
                prediction = predict_lgc(predictor, trial, sigma, args.alpha)
                metrics = evaluate(trial, prediction)
                rows.append({"seed": seed, "method": "LGC", "sigma": sigma,
                             "alpha": args.alpha, **metrics})
                predictions.append((seed, "LGC", sigma, prediction))
                paired.append({"seed": seed, "sigma": sigma, "alpha": args.alpha,
                               "lgc_minus_1nn_error": metrics["error_rate"] - baseline_metrics["error_rate"]})

    # No output is created until every predictor call has completed successfully.
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "trials.csv", rows)
    summary = summarize(rows)
    write_csv(args.out / "summary.csv", summary)
    if paired:
        write_csv(args.out / "paired.csv", paired)
    else:
        # Avoid retaining a stale comparison when a directory is reused.
        (args.out / "paired.csv").unlink(missing_ok=True)
    point_rows = [
        {"seed": seed, "index": i, "x0": float(point[0]), "x1": float(point[1]),
         "observed_label": int(trial.observed[i]), "true_label": int(trial.truth[i])}
        for seed, trial in zip(args.seeds, trials) for i, point in enumerate(trial.X)
    ]
    write_csv(args.out / "points.csv", point_rows)
    write_csv(args.out / "predictions.csv", [
        {"seed": seed, "method": method, "sigma": sigma, "index": i, "prediction": int(label)}
        for seed, method, sigma, labels in predictions for i, label in enumerate(labels)
    ])
    manifest = {
        "status": "core comparison run; review required" if predictor else "baseline only; LGC pending",
        "reproduction": "partial qualitative reproduction; newly generated toy data",
        "seeds": args.seeds, "n_samples": args.n_samples, "noise": args.noise,
        "labels_per_class": 1, "evaluation": "unrevealed labels only",
        "geometry": "evenly spaced semicircles + independent Gaussian coordinate noise; shuffled",
        "rng": "NumPy default_rng; SeedSequence(seed).spawn(2) for data and labels",
        "predictor": args.predictor, "sigmas": args.sigmas if predictor else [],
        "alpha": args.alpha if predictor else None,
        "parameter_policy": "fixed defaults / descriptive sweep; no selection using evaluation labels",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "standard_deviation": "sample SD across seeds (ddof=1); not a confidence interval",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for row in summary:
        print(f"{row['method']} sigma={row['sigma'] or 'n/a'}: error "
              f"{100 * row['mean_error']:.2f}% +/- {100 * row['std_error']:.2f} pp (sample SD)")
    print(manifest["status"])


if __name__ == "__main__":
    main()
