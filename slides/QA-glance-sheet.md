# Q&A glance sheet — Dylan (B), Group 28, label propagation
Short lines. Left = what they ask. Right = what you say.

## The one paragraph that answers most things
Normalise the affinity so eigenvalues sit in [−1, 1]. Multiply by α < 1 → contraction. So the loop must settle, the inverse exists, the answer is unique, and the same answer is the minimum of smoothness + fidelity.

## Eigenvalues — why [−1, 1]
- Row-stochastic P = D⁻¹W: each row = 100% of a point split among neighbours. Distributes, never multiplies → can't stretch past 1.
- S = D^(-1/2) W D^(-1/2) = D^(1/2) P D^(-1/2). Rescale before, undo after → **same eigenvalues as P** (similarity).
- S symmetric → eigenvalues all **real**, so no imaginary parts, no oscillation.
- Top eigenvalue = exactly **1**, and ‖S‖₂ = 1.0 measured. We verified.
- α = 0.99 → every direction shrinks ≥1% per round.

## Multi-hop / the expansion
- Geometric series, **not** binomial: (I − αS)⁻¹ = I + αS + α²S² + …
- S²[i][j] = Σₖ S[i][k]S[k][j] → matrix multiply IS path counting. A–B–C example: S[A][C]=0, S²[A][C]≠0.
- α^k discounts long paths. At 0.99: 10 hops ≈ 0.90, 100 ≈ 0.37, 300 ≈ 0.05.
- Infinite terms, **finite sum** — say both halves.

## Local vs global
- Local = one hop (k=1) = neighbours agree. That's all 1-NN can do.
- Global = long chains (k big) = same crescent agrees. That's the tail of the series.
- Can't truncate → can't have local without global. That IS global consistency.

## α — where it comes from
- α = trust dial in the loop: new = α(neighbours) + (1−α)(original label).
- Same dial in the energy as μ: **μ = (1−α)/α**. Never "μ = α".
- Paper uses α = 0.99 on digits/text.
- Small α → stubborn (≈1-NN, local only). Large α → gullible, labels flow everywhere.

## The two ends of one equation
- Iterate = solve = minimise. Three names, one object.
- Closed form = how you compute. Objective = why it's right + how you'd extend it.
- Convex bowl (μ > 0) → strictly convex → **unique** minimum.
- Verified: stationarity 2.1e-17, 18/18 perturbations raised Q.

## Roughness / stubbornness, visually
- Roughness = count edges joining different colours, weighted by strength, normalised by degree (so an isolated point can't win by being alone).
- Stubbornness = how far current scores drifted from the 2 known labels. Zero only at revealed points.

## Verification numbers (slide 7)
- Loop vs direct solve: agree ~**1e-10**, worst 8×10⁻¹¹. Zero label mismatches, all 10 seeds, 400 points.
- Stopping rule: residual underreports true error by **1/(1−α)** → 100× at α=0.99. Predicted 1e-10, measured 9.1e-11. Predicted 1e-9 at 0.999, measured 7.5e-10.
- Iters: 1,405 actual at α=0.99 vs bound 2,750 → bound holds, is loose.
- Team result: **0.45% vs 22.36%** (1-NN), σ=0.15, 10 fixed seeds.

## The identity audit (have it ready, don't lead with it)
- False as matrix algebra, harmless for classification. Missing D → per-row scaling → argmax unchanged.
- Gaps: printed 0.82 (broken), corrected form 1.4e-15, ≤6/200 predictions shift.
- If pressed: "we checked every published identity; this is the one that's wrong."

## Question bank (asked → answer)
- Why zero diagonal in W? → self-reinforcement.
- Disconnected graph? → multiple λ=1 modes; labels can't cross components.
- Why does (I − αS) exist? → eigenvalues of I − αS in [1−α, 1+α], never 0.
- Why symmetric S not P? → symmetric = real spectrum; the Eq.6 derivation is for S.
- Inductive / unseen points? → NOT tested. Transductive only: features yes, labels never.
- Replicated the paper? → No. Toy two-moons, mechanism only, our own baseline.
- σ=0.15 = paper's setting? → No, ours, fixed before the run. Paper's 0.15 is a *text* bandwidth.
- Wins every trial? → Only σ ≤ 0.30. At σ=0.60: 32.59%, 1-NN wins 8/10.
- Harder problem? → Choosing σ with 2 labels. No honest way to validate.
- Scale? → dense W is O(n²); at n=10⁵ ≈ 80 GB. Fix: sparse k-NN graph + conjugate gradient.
- Is SD a confidence interval? → No. Across seeds only.

## Never say
- F\* without the (1−α) — constant scales all rows equally, argmax unchanged (paper's Eq. 3 omits it, that's fine).
- "converges because α < 1" alone → it's **eigenvalue bound + α < 1**.
- "binomial expansion" → geometric series.
- "we replicated the paper's results".
- Anything about inductive behaviour or the Figure-4 extension — we didn't implement it.
- No claim about the paper's Figure 5 error bars (it has none).

## Blanked? three sentences carry slide 5
1. Normalisation puts S's eigenvalues in [−1, 1], so with α < 1 the loop is a contraction — it must settle.
2. When it settles, F\* = (1−α)(I − αS)⁻¹Y, and the inverse exists because it's a contraction.
3. That inverse is a diffusion kernel — α^k weights paths of length k — and the same F\* minimises smoothness + fidelity: ridge regression with a graph.
