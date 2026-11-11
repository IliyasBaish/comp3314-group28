# Paired two-moons comparison

This run uses Iliyas’s NumPy predictor from commit `645b7b8`. Each method gets
the same 400 points and two revealed labels per seed. We score the remaining
398 points across ten fixed seeds. LGC uses α = 0.99 throughout.

| Method | σ | Mean error | Sample SD (percentage points) |
|---|---:|---:|---:|
| 1-NN | — | 22.36% | 4.22 |
| LGC, fixed comparison | 0.15 | 0.45% | 0.54 |
| LGC | 0.05 | 0.15% | 0.40 |
| LGC | 0.10 | 0.10% | 0.13 |
| LGC | 0.30 | 14.07% | 5.21 |
| LGC | 0.60 | 32.59% | 13.20 |

At σ = 0.15, LGC improved on 1-NN in all ten trials and matched Dylan’s
per-seed reference accuracies. The other bandwidths describe sensitivity on
the same trials. The fixed comparison uses the bandwidth specified before
the run. The results concern generated two-moons data with one label per class.

[Seed 0 predictions](example.png) show the fixed comparison. [Error summary](error_summary.png)
shows all settings with sample SD. The [experiment guide](../../experiments/README.md)
contains the run commands. See [per-trial metrics](trials.csv), [paired differences](paired.csv),
and the [run configuration](manifest.json) for details.
