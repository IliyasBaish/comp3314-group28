# Original 1-NN baseline run

The [paired experiment](../comparison/README.md) uses these same trials and
adds predictions from the team's LGC implementation.

Generated using `python -m experiments.run` and `python -m experiments.plot`.
See `manifest.json` for the exact environment and protocol, and
`experiments/README.md` for setup, commands, and output schemas.

| Method | Trials | Mean unrevealed-point error | Sample SD |
|---|---|---|---|
| Euclidean 1-NN | 10 | 22.36% | 4.22 percentage points |
| Team's LGC implementation | Not run | Pending | Pending |

Each run has 400 generated points, exactly two revealed labels (one per class),
and 398 scored points. Seed 0 is the preselected example figure. Mean accuracy
is 77.64%. The sample SD describes variation across runs, not a confidence interval.

The full hidden truth is present in `points.csv` solely for audit and scoring.
It must not be passed to a predictor. No LGC result is present, no best seed
was selected, and no claim of superiority follows from the baseline alone.

The generated figures and CSVs are our own toy experiment, not figures/results
copied from Zhou et al. This is preparation for a partial qualitative reproduction.
