# B's theory slides — draft content (2 min, speaker: Dylan)

For D to assemble. Two slides plus a back-up slide for Q&A. All equations cross-checked
against the PDF (Section 2 = algorithm + convergence, Section 3 = regularization;
our README's "Section 3" scope covers both because the convergence proof lives in §2).

## Slide B1 — Why the iteration has a closed form

**Content:**
- Iteration: `F(t+1) = α S F(t) + (1−α) Y`, with `S = D^-1/2 W D^-1/2`, W the zero-diagonal RBF affinity
- Two facts make convergence automatic:
  - S is **similar to the row-stochastic matrix** `D⁻¹W` → same eigenvalues, and S is symmetric → all real, all in [−1, 1] (‖S‖₂ = 1)
  - so `αS` is a **contraction** (α < 1) and `Σ (αS)ⁱ = (I − αS)⁻¹` (Neumann series)
- Therefore `F* = (I − αS)⁻¹ Y` (their Eq. 3), reached from any starting point
- `(I − αS)⁻¹` **is a diffusion kernel** — labels spread along every path in the graph, αᵏ weights path length k. That's "global consistency" made explicit.

**Speaker notes (≈45 s):** The algorithm looks like a heuristic, but the normalization step (2) is the quiet hero: it puts S's spectrum in [−1, 1], which makes αS contract, which gives a closed form. Same trick as PageRank's damping factor. Also worth one sentence: symmetric normalization matches spectral clustering, the paper itself says steps 1–2 are identical to NGJ (ref [9]).

## Slide B2 — It's also the exact minimizer of a regularized objective

**Content:**
- `Q(F) = ½ Σᵢⱼ Wᵢⱼ ‖Fᵢ/√Dᵢᵢ − Fⱼ/√Dⱼⱼ‖² + μ Σᵢ ‖Fᵢ − Yᵢ‖²` (their Eq. 4)
- 1st term = **smoothness** (graph Laplacian energy, weighted so no hub dominates), 2nd = **fitting** (stay near clamped labels). μ trades them off.
- Setting ∂Q/∂F = 0: `F* − SF* + μ(F* − Y) = 0`, substitute `α = 1/(1+μ)` → `(I − αS)F* = (1−α)Y` — **the same fixed point** (their Eq. 6)
- So: iterate = solve = minimize. Three views of one object, and Q is convex quadratic so the solution is unique.

**Speaker notes (≈45 s):** This is our course's ridge-regression story in graph clothing: minimize (roughness penalty) + (fidelity), derivative = 0 gives a linear system. μ ↔ α dictionary: μ = (1−α)/α. One honest gap for the report, not the slides: §2's F* from iteration equals §3's Q-minimizer only up to the constant (1−α); argmax classification is unchanged. Also mention we verified this numerically (bonus check, B3) — iteration matched np.linalg.solve to ~1e-10 and stopped exactly where theory says it should.

## Back-up B3 — Our convergence check (only if asked / if C's results slide needs support)

- Two-moons toy, 1 label per class, α sweep {0.5 … 0.999}
- Iteration vs direct solve: agreement 1e-12…1e-9, zero argmax mismatches
- Measured error = tol/(1−α), because the observable residual criterion carries an extra (1−α) factor
- Unlabeled accuracy rises with α (0.77 → 1.0) — that knob IS the local/global trade-off
- See `results/convergence_check_summary.md`

## Course connections (mix into B1/B2 verbally, don't crowd slides)

| Paper idea | Where it appears in COMP3314 |
|---|---|
| Regularized objective + stationarity | Ridge regression / least-squares normal equations (PS1 Q3, Q7 Adaline gradient steps) |
| Kernel affinity W = RBF | SVM with RBF kernel the paper beats in Fig 1(b) (PS1 Q4/Q7) |
| Fixed-point iteration `F ← αSF + βY` | PageRank-style power iteration; gradient descent as a contraction |
| Spectral normalization `D^-1/2 W D^-1/2` | Spectral clustering (NGJ), same Laplacian geometry as PCA eigengap talk |
| Error rate only on unlabeled points | Transductive evaluation vs inductive test sets |
| k-NN / 1-NN baselines | PS1 Q6-ish intuition: supervised only uses *local* consistency |

## Q&A prep for B

- "Why must W have zero diagonal?" → prevents self-reinforcement, the paper's own remark after the algorithm box
- "What if the graph is disconnected?" → D⁻¹W has multiple λ=1 modes; labels can't cross components, propagation is per-component (harmonic function intuition). Our toy with big σ keeps one component.
- "What does α actually control?" → diffusion distance before damping kills it; α→1 pure harmonic (labels reach everywhere), α→0 clamped-only (1-NN-ish local). Our sweep shows this as accuracy 0.77→1.00.
- "Why symmetric S and not row-stochastic P?" → they differ; paper tests both variants (§2 last para, variant (1) uses P; Eq. (6) derivation is for S). Symmetric is what makes the spectrum real.
- "Is (I−αS) always invertible?" → yes for α<1 since |λ(S)|≤1 ⇒ eigenvalues of I−αS in [1−α, 1+α], bounded away from 0.
