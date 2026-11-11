# Convergence check — Role B bonus (our results, not the paper's)

Script: `experiments/convergence_check.py` (numpy only; run `python3 -m experiments.convergence_check ...` from the repo root, commands in the file header).
Data: 200-point two-moons toy, noise 0.1, seed 0, one revealed label per class
(the paper's Figure 1(a) pattern; the graph construction is exactly Section 2:
zero-diagonal RBF affinity with σ = 0.1, symmetric normalisation `S = D^-1/2 W D^-1/2`).
Paper uses α = 0.99, we sweep α so the convergence-rate story is visible.

## Claim under test

Section 2 says iterating `F(t+1) = α S F(t) + (1−α) Y` converges to the closed form,
and Section 3 (Eq. 6) says that closed form is the exact minimiser of the regularised
objective `Q(F)`. Numerically that means: **iterating, and solving the linear system
`(I − αS) F = (1−α) Y` directly, must give the same F** — and the convergence proof's
ingredients (eigenvalues of S in [−1, 1], error decaying like αᵗ, independence from F(0))
should be observable.

## Results (`convergence_check.csv`)

| α | eigmax\|S\| | iters (residual < 1e-12) | predicted iters ln(tol)/ln(α) | ‖F_iter − F_solve‖∞ | slope ln α | observed slope | F(0)-independence | label mismatches | unlabeled acc |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 1.0 | 29 | 40 | 8.1e-13 | −0.693 | n/a* | 1.7e-12 | 0 | 0.77 |
| 0.9 | 1.0 | 167 | 263 | 8.2e-12 | −0.105 | −0.130 | 1.6e-11 | 0 | 0.88 |
| 0.95 | 1.0 | 326 | 539 | 1.7e-11 | −0.051 | −0.062 | 3.5e-11 | 0 | 0.94 |
| 0.99 | 1.0 | 1405 | 2750 | 9.1e-11 | −0.010 | −0.013 | 1.9e-10 | 0 | 0.97 |
| 0.999 | 1.0 | 10860 | 27618 | 7.5e-10 | −0.001 | −0.0014 | 1.7e-09 | 0 | 1.00 |

\* α = 0.5 converges inside 29 steps, too few in the log-error fit window to measure a slope.

## What held

1. **Iteration == linear solve.** Max-abs agreement is 1e-12…1e-9, and both argmax
   classifications are identical for every α. The paper's Eq. (3)/(6) equivalence is real.
2. **Spectral bound.** ‖S‖₂ = 1.0 exactly at every α, matching the paper's claim that
   S is similar to the row-stochastic `D⁻¹W` (same eigenvalues, symmetric so all real).
3. **No dependence on F(0).** Starting from Y, from zeros, and from a random matrix
   lands on the same limit to ~1e-9.

## Honest nuances worth putting in the report

- **We stop on the residual, not the error.** The script stops when
  `‖F(t) − F(t−1)‖∞ < 1e-12`, which is the only thing a user can observe. But
  `F(t+1) − F(t) = (αS)^t (I − αS)(F(0) − F*)`, and `(I − αS)` shrinks the top
  eigenmode by ≈ (1−α). So residual < tol means **error ≈ tol/(1−α)**, and that is
  exactly what we measure: 1e-12/(1−0.99) = 1e-10 vs observed 9.1e-11, and
  1e-12/(1−0.999) = 1e-9 vs observed 7.5e-10. The theory predicts its own stopping error.
- **Observed decay is slightly faster than αᵗ** (slope −0.0132 vs ln 0.99 = −0.0101).
  The spectral radius only bounds the worst case — the αᵗ term rides on the λ = 1
  eigenvector (degree-weighted `D^1/2 1`), and for these starting conditions that
  component nearly cancels in the max norm, so the next eigenvalue dominates.
- **Accuracy grows with α** (0.77 → 1.00), which is the local-vs-global-consistency
  knob: bigger α diffuses labels further before stopping. Cost: ~10,000 iterations at
  0.999 vs 1,405 at 0.99. On this toy the paper's fixed 0.99 is cheap enough that
  0.999 is reachable anyway.

## Follow-up pass (later the same day, after Role C landed)

- **Variants audit** (`results/variants_audit.csv`): §2's identity
  `(I−αP)⁻¹Y == (D−αW)⁻¹DY` holds to 1.6e-15. But §2's claim that the transpose
  variant "is equivalent to F* = (D−αW)⁻¹DY" fails literally (relative gap 0.82);
  the algebra-correct form `(I−αPᵀ)⁻¹Y = D·(D−αW)⁻¹Y` verifies to 2.5e-15.
  Pre-multiplying by D is a positive per-row rescaling, so the **argmax rule is
  unaffected** — "equivalent for classification" is defensible while the printed
  matrix identity is not. All variants differ from the main S-based argmax at
  ≤6/200 toy points: a robustness observation, and an equation-check finding.
- **Q(F) optimality (§3 Eq. 4-6):** stationarity residual 2.1e-17 at F\* with
  μ = (1−α)/α, and 18/18 random perturbations of F\* raised Q. Solid.
- **Protocol mode** (`--protocol`, `results/protocol_check.csv`): on C's exact
  trials (seeds 0-9, n=400, noise 0.08, σ=0.15, α=0.99) every trial converges to
  the solve within 8.3e-11 (= tol/(1−α), theory again), zero argmax mismatches,
  and the reference propagation accuracy averages **99.55%** against his 1-NN
  accuracy of 77.64% (`results/baseline/summary.csv`). This is B's reference
  pending A's core, **not a team comparison claim** until `src.lgc:predict`
  reproduces it.
- **Plot:** `results/convergence_rate.png`, generated with
  `uv run --with matplotlib python3 -m experiments.convergence_check --sweep 0.5 0.9 0.95 0.99 0.999 --plot`
  (matplotlib deliberately not a repo requirement).

## Caveats

σ = 0.1 and the two revealed points are our choices (paper leaves the toy bandwidth
unstated), so this is a qualitative reproduction of Figure 2, not a replication of
their numbers. Bandwidth sensitivity is Role C's experiment, deliberately not covered here.
