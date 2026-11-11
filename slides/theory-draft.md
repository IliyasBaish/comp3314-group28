# B's theory slides (2 min, speaker: Dylan)

Format matched to `slides/evidence-draft.md` so D can assemble both the same way.
Equations and numbers all cross-checked against the PDF and `results/` (see
`docs/b_review_of_c.md` and `results/convergence_check_summary.md`).

## Slide 1 — The quiet hero is the normalization (about 60 seconds)

**Visual:** Paper Figure 2 (convergence panels t=10 to t=400), credited to Zhou
et al., NIPS 2003. Optionally the update rule from the Section 2 algorithm box,
re-typed, not screenshotted.

- `S = D^-1/2 W D^-1/2` is similar to the random-walk matrix `D⁻¹W` → eigenvalues all in [−1, 1], exactly one at the top
- α < 1 makes `αS` a contraction → iterates converge from any start, to the closed form `F* = (I − αS)⁻¹ Y`
- `Σ (αS)ⁱ = (I − αS)⁻¹` is a diffusion kernel: a label reaches every point along all paths, weighted αᵏᵗʰʳʷᵉʳᵉ = path length → *global* consistency, literally
- Same shape as PageRank: α is the damping factor, Y the teleport vector; steps 1–2 are also spectral clustering's construction (paper cites Ng-Jordan-Weiss)

**Say:** "The algorithm looks like a heuristic. It isn't. Step two normalizes the
similarity graph, and that quiet step is the whole ballgame. The normalized matrix
is similar to a random walk matrix, so its eigenvalues live between minus one and
one, and when alpha is below one, every pass shrinks the previous state. So the
iteration converges no matter where you start, to a closed form, and because the
sum of matrix powers telescopes to the inverse, that closed form is a diffusion
kernel. One label reaches every point along every path in the graph, with alpha
killing long paths. That's the global in global consistency. If this shape feels
familiar, it's PageRank, same damping factor, same teleport idea. And the first
two steps are exactly spectral clustering, so the graph machinery from our
clustering lectures shows up inside a classifier."

## Slide 2 — Ridge regression wearing graph clothing (about 60 seconds)

**Visual:** `../results/convergence_rate.png` (our figure, label "our results") —
straight log-error lines per α, plus the dashed predicted slope.

- `Q(F) = ½ Σ W_ij‖F_i/√d_i − F_j/√d_j‖² + μ Σ‖F_i − Y_i‖²` — smoothness + fitting, μ is the dial, α = 1/(1+μ)
- ∂Q/∂F = 0 **is** the fixed point: iterate = solve a linear system = minimize a convex quadratic. Three views, one object, unique answer
- Our verification: stationarity residual 2e-17 at F\*, every random perturbation raised Q (18/18), stopping error equals tol/(1−α) exactly (9.1e-11 observed vs 1e-10 predicted at α=0.99), decay slope within a whisker of the predicted ln α
- Course connection to say, not to print: same normal-equations structure as the least-squares and Adaline questions in PS1, just with the graph Laplacian replacing the weight penalty

**Say:** "Why should that fixed point be the right answer? Because it's the exact
minimizer of an honest objective. Half the graph smoothness energy, plus mu times
the squared distance to the labels we actually know. First term says predictions
should change slowly along edges, second says don't drift from the clamped labels.
That's the ridge regression story from every regularization lecture, with the
graph Laplacian in place of the weight penalty, and mu trading them off. Take the
derivative, set it to zero, substitute one over one plus mu, and the paper's
iteration falls out. So iterating, solving a linear system, and minimizing an
objective are the same thing. And we didn't take that on faith. On our data the
gradient at the claimed minimum is zero to machine precision, every random nudge
we tried made the objective worse, and the error curve falls exactly as fast as
the theory says, with the observed tolerance matching the prediction to one
significant figure."

## Backup slide — The equation check (only if asked, or if we steal 20 s)

- Paper's last §2 paragraph claims the transpose variant is "equivalent to (D − αW)⁻¹DY". As matrices it isn't (measured relative gap 0.82); the true identity is `(I − αPᵀ)⁻¹Y = D·(D − αW)⁻¹Y`, verified to 2.5e-15
- Pre-multiplying by D is a positive row rescaling → argmax unchanged → "equivalent" survives as a statement about classifications, not equations
- All operator variants (S-based, P, Pᵀ, with or without DY) shift at most 6/200 toy predictions: classification is robust to the normalization choice
- B's independent reference (0/400 vs A's core, all seeds) is the integration receipt: `results/protocol_check.csv`

## Q&A prep for B

- "Why zero diagonal in W?" → self-reinforcement would let a point dominate its own label forever; paper's own remark after the algorithm box
- "What if the graph is disconnected?" → `D⁻¹W` then has multiple λ=1 modes, labels cannot cross components; propagation is per-component (harmonic intuition). Wide-σ sweeps in our results walk into this
- "What does α really control?" → diffusion reach before damping kills it: α→1 pure harmonic/global, α→0 clamped-only/local. Our sweep shows it as accuracy 0.77 → 1.00 (standalone toy) and the σ sweep shows the complementary knob
- "Why symmetric S and not row-stochastic P?" → symmetric ⟹ real spectrum, clean convergence proof; the paper tests the P variants and the experiment section's variant (2) even drops the D factor on Y (audited in backup slide)
- "Is I−αS always invertible?" → yes for α<1: eigenvalues of I−αS lie in [1−α, 1+α], bounded away from zero
- "Your error bars?" → we only run toy-scale experiments (400 points, dense solve O(n³)); the dense-matrix cubic cost is our complexity observation, not a paper measurement

## Deck-level notes for D

- B's segment sits between A's algorithm and C's evidence (plan's run order); both main slides reuse figures A/C may also want, so keep the credit lines distinct (Figure 2 = paper, convergence_rate.png = ours)
- 2:00 total: slide 1 script ≈ 65 s, slide 2 ≈ 60 s — trim the PageRank sentence first if rehearsing long
