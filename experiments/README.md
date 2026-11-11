# Role C: experiment protocol and handoff

Owner: Surkhay. Core LGC algorithm: Iliyas (A). Equation/convergence checks: B.
Bandwidth analysis and final slide assembly: D. This folder implements C's work
without supplying or claiming to validate A's missing core algorithm.

## Setup and current run

From the repository root (tested with Python 3.14.6):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-experiments.txt
python -m unittest discover -s tests -v
python -m experiments.run
python -m experiments.plot
```

The default output is `results/baseline/`. It is **1-NN only**. No LGC results,
improvement claim, or completed from-scratch reproduction is implied.
The manifest records the actual Python and NumPy versions used.

## Fixed protocol

- Seeds 0 through 9, each generating a fresh, balanced 400-point dataset.
- Two semicircles: `(cos(t), sin(t))` and `(1-cos(t), 0.5-sin(t))`,
  with evenly spaced `t` in `[0, pi]`, then independent Gaussian coordinate
  noise of standard deviation 0.08. Shuffle sample order.
- Separate RNG streams for data generation and label selection.
- Uniformly sample exactly one revealed point per class, leaving 398 for evaluation.
- A predictor sees all coordinates, but only the two revealed labels; all others
  are encoded as -1. The full truth is available only to evaluation/output code.
- 1-NN uses Euclidean distance to the two revealed points. No feature scaling.
- Every method/bandwidth receives the same dataset and label placement for each seed.
- Error = incorrect unrevealed predictions / 398. Report the mean and sample SD
  (`ddof=1`) across seeds. The SD is not a confidence interval or standard error.
- Default proposed LGC configuration: sigma=0.15 and alpha=0.99. Sigma is an
  initial fixed toy-data configuration, **not** a demonstrated optimal value or
  a verified copy of the paper's toy setting. Alpha follows Section 4.
- Do not choose a bandwidth or seed by evaluation error. For D's descriptive
  bandwidth sweep, show all declared settings and retain all trials.
- Seed 0 is the predetermined illustration. It is not selected for visual success.

The 10 runs vary both data/noise and label placement. This does not isolate those
two sources of variation and is not evidence of general performance on real data.
Having a label from every class is a favorable sampling assumption, not guaranteed
when labels arrive naturally. This is a partial qualitative reproduction of the
two-moons idea, not a replication of the paper's samples, full benchmarks, or numbers.

## Proposed interface for A (agree before integration)

Provide a callable accessible as `module:function`:

```python
def predict(X, observed_labels, *, sigma, alpha):
    # X: float array, shape (n, 2); all coordinates are available.
    # observed_labels: integer array, shape (n,); -1 means unrevealed, 0/1 known.
    # Return integer predictions of shape (n,), each 0 or 1.
    ...
```

This interface is proposed locally; it has not been agreed with Iliyas yet.
If his core returns an `(n, 2)` score matrix, a small wrapper can return
`scores.argmax(axis=1)`. Do not pass the true-label CSV into the predictor.
Use sigma in `exp(-squared_distance / (2*sigma**2))`; agree on this convention
to avoid confusing sigma with gamma. Use zero diagonal and symmetric degree
normalization as in Section 2. B should independently check the fixed point.
For a numerically isolated vertex, agree on an explicit handling policy instead
of producing NaNs. Make non-convergence an explicit error or diagnostic.

Once the actual callable exists (the following module is an **example path**):

```bash
python -m experiments.run --predictor src.lgc:predict --out results/comparison
python -m experiments.plot --results results/comparison
```

The runner does not silently fall back to another implementation. Import errors,
invalid class predictions, and exceptions stop the run before new CSVs are written.
The adapter checks output shape/labels; it does not establish algorithm correctness.
Use a fresh output directory for a different experiment configuration.

## Handoff to D: sensitivity results

After integration, D can run the same fixed seeds at a declared bandwidth grid:

```bash
python -m experiments.run --predictor src.lgc:predict --sigmas 0.05 0.1 0.15 0.3 0.6 --out results/sensitivity
```

The grid is proposed, not yet run. Keep alpha and the label budget fixed. A plot
of error versus sigma should include mean and sample SD, with the unchanged
1-NN mean as a reference. Do not present the lowest-error sigma as an unbiased
test result; selecting it would require separate validation labels or a separate
development protocol. D owns the final sensitivity interpretation and plot.

## Outputs

| File | Meaning |
|---|---|
| `trials.csv` | Per-seed/method metrics; sigma/alpha blank for 1-NN |
| `summary.csv` | Mean error, sample SD, mean accuracy for each configuration |
| `paired.csv` | Only when core runs: LGC minus 1-NN error for each seed/bandwidth; negative favors LGC |
| `points.csv` | Coordinates, revealed-label mask, and truth for audit; not predictor input |
| `predictions.csv` | Every prediction, allowing metrics to be independently recomputed |
| `manifest.json` | Configuration, versions, RNG policy, and pending/completed-run status |
| `example.png` | First seed's input and recorded predictions; revealed points are stars |
| `error_summary.png` | Mean error with sample SD; no pending method is plotted |

Data generation, the Euclidean baseline, and evaluation are written directly with
NumPy. NumPy and Matplotlib are credited dependencies. No dataset or third-party
algorithm implementation is bundled. Original method/reference:
[Zhou et al., NIPS 2003](https://proceedings.neurips.cc/paper_files/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf),
Sections 1, 2, and 4. New code and draft prose were prepared with Codex assistance;
Surkhay and the team should review, understand, and follow course disclosure rules.
