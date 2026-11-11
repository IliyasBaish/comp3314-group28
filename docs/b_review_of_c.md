# B reviews C — Role C deliverables (965eda8)

Reviewer: Dylan (B). Scope: `docs/c_*`, `experiments/{protocol,run,plot}.py`, `tests/`,
`results/baseline/`, `report/c_evidence_draft.tex`. Reviewed 11 Nov, Python 3.14.7 /
numpy 2.3.5 on my machine (yours: 3.14.6 / 2.5.1).

## Verdict: ship it. Everything checkable is verified.

## Numerics — reproduced exactly

- `python3 -m unittest discover -s tests` → 9 tests OK.
- Reran `python3 -m experiments.run` into a fresh directory: per-seed error counts
  **identical on all 10 seeds** and summary matches (1-NN mean 22.36%, sample SD
  4.22pp). Bit-identical across numpy 2.3.5 and 2.5.1, so PCG64 + SeedSequence.spawn
  streams are stable, integration on A's machine should reproduce too.

## Paper claims — checked against the full PDF

Every Section 4 number in `c_paper_evidence.md` matches the paper: USPS digits 1-4,
16x16, counts 1,269/929/824/852 = 3,874; k-NN k=1; digit RBF widths SVM 5, harmonic
and consistency 1.25, zero diagonal; text = `rec` topics from 20-news-18828 (autos,
motorcycles, baseball, hockey), Rainbow, Porter stemmer before counting, SMART
stoplist, headers skipped, words in ≤5 docs dropped, empty docs removed → 3,970 docs
× 8,014 dims, TF-IDF; distance 1−cosine; text widths SVM 1.5, harmonic/consistency
0.15; 100 trials; ≥1 label per class by construction; α=0.99; jitter kernel ≈1%
error at 30 labels on digits. The harmonic method's nonmonotone text behaviour is
correctly attributed as the **authors'** explanation (§4.3), and the "don't eyeball
Figure 5 into percentages" warning is right — I have nothing to add there.

## Protocol soundness — no leakage found

Checked the actual code, not just the README: data and label RNG streams are
separated before anything sees the labels; predictors get only `observed` with −1
masking and copies; truth lives only in `evaluate`/output; seed 0 for the example
is declared upfront, not chosen; the runner writes nothing until every predictor
call succeeds, no silent fallback. The σ convention line (exp(−d²/2σ²), not gamma)
is exactly what integration day will need — I match it.

## The thing you asked B to do: fixed point — done

`python3 -m experiments.convergence_check --protocol` (σ=0.15, α=0.99 on your exact
seeds 0-9): all trials converge to the closed-form solve within 8.3e-11 (consistent
with the tol/(1−α) prediction), zero argmax mismatches, and the reference
propagation accuracy is **99.55% mean** vs your 1-NN 77.64%. Per-seed table in
`results/protocol_check.csv`. Treat that as the expected value for A's core: when
his `src.lgc:predict` lands, `python -m experiments.run --predictor src.lgc:predict`
should reproduce ~99.5%, and any deviation is a flag, not a discovery. Marked
clearly in the CSV as B's reference pending A, so our report comparison claim only
fires once both implementations agree. **(They agreed the same night: PR #3 merged,
0/400 label differences per seed — see addendum.)**

## Two nits

1. `plot.py` example panel: `manifest["sigmas"][0]` will KeyError if a predictor run
   somehow has empty sigmas — currently guarded by run.py's `sigmas if predictor
   else []`, fine, just fragile if someone hand-writes manifests.
2. `c_evidence_draft.tex` says "Figure~5 of the original paper" — when this gets
   integrated we need to decide whether we include the figure (copyright: paper
   figures shouldn't be pasted into our report; refer instead). I'll flag to D when
   the figure policy for slides comes up (slides label "paper figures vs ours", so
   one figure there is probably fine, but report stays prose).

## Post-merge addendum (same night — C's comparison in fe0d47c, re-reviewed)

- Reproduced `experiments.run --predictor src.lgc:predict --sigmas 0.05 0.1 0.15 0.3 0.6`
  on merged main: **all 60 trial rows reproduce exactly** (same numpy lineages, same seeds).
- Recomputed the headline metrics from `results/comparison/predictions.csv` + `points.csv`
  as promised in the protocol: 890/3980 1-NN misclassifications = 22.36% ✓.
- All ten sigma=0.15 paired diffs negative (worst -16.8pp) ✓. One scope nit for the report
  line before the final freeze: "lower error in every paired trial" is true *at sigma=0.15*,
  but `paired.csv` has 50 rows and eight of them (all sigma=0.60) favor 1-NN by up to
  +26.9pp — suggest the draft's sentence explicitly scopes to the primary config so nobody
  reads it across the sweep. The sweep paragraph itself already tells the truth here,
  so this is one clause of wording, not a content fix.
- `report/main.tex` recompiles clean with the updated `c_evidence_draft.tex` (3 pages,
  the new `Section~\ref{sec:theory}` reference resolves against my theory section ✓).

## Review-chain note for the others

This satisfies "B reviews C" for the 12-Nov checkpoint (experiments validated,
evidence numbers checked). The remaining review edge: D reviews A once his core and
slides exist.
