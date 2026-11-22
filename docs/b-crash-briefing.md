# Plain-language crash briefing — what Group 28's project actually says

Read this once, top to bottom, ~10 minutes. Every number here is from the merged
`results/*.csv` and `report/`, so it's safe to repeat.

## The whole thing in one line
You have **2 labelled points and 398 unlabelled ones**, and the question is whether
the *shape* of the unlabelled points alone can tell you where the boundary is.

## The trick
1. Wire the data together: connect every pair of points with a strength
   `W_ij = exp(-‖x_i - x_j‖² / 2σ²)`. Close points get a strong wire, far points
   almost none. **σ is the only judgement call** — it decides what counts as "close".
   (`W_ii = 0`: a point must not vote for itself.)
2. Bleed the labels along the wires: `F(t+1) = α·S·F(t) + (1−α)·Y`, where `S` is `W`
   normalised at both ends (`D^-1/2 W D^-1/2`) and `Y` holds the 2 known labels.
   Each round, every point's score is pulled toward its neighbours' scores, then the
   true labels are pushed back in so they never drift.
3. Repeat until nothing changes; the surviving score per point is its class.

**Why it works on two moons:** the two ends of the same crescent are far apart in a
straight line but *close along the crescent* — so the wire path carries one label
around the whole arc. 1-nearest-neighbour can only see the straight line, so it fails.
That single sentence is the answer to "what problem does this solve that 1-NN can't".

## Why it can't fail (B's part, and the bit worth understanding)
The normalisation in step 2 makes `S` a rescaled copy of a "who-do-I-tell" matrix, so
its eigenvalues land in **[−1, 1]**, with λ=1 present (and only one λ=1 mode if the
graph is connected — that's the same fact behind the "what if the graph is disconnected?"
answer: several λ=1 modes, and labels can't cross between components). `‖S‖₂ = 1`, we
measured it.
Multiply by α < 1 and every round *shrinks* the remaining difference — a contraction,
like a damped echo. So: one stable answer, reached **from any starting point**, and it
has a closed form, `F* = (1−α)(I − αS)⁻¹ Y`. The `(I − αS)⁻¹` piece is literally a
**diffusion kernel**: expand it as `I + αS + α²S² + …` and the term α^k counts label
flow along paths of length k — that's "global consistency" written as algebra. α is the
dial: α→0 stays local (just your own label), α→1 diffuses everywhere (harmonic/global).
PageRank uses the same damping trick, and the same normalised matrix is what spectral
clustering uses — that's the "connects to this course" answer.

## The other view of the same thing (Q)
That fixed point is also the *unique* minimum of an energy: **(how rough the labels are
across wires) + μ·(how far you drifted from the true labels)**. Set its slope to zero
and you get exactly the iteration's answer, with `μ = (1−α)/α`. This is ridge regression
wearing a graph: instead of penalising big weights you penalise big differences between
neighbours. Because it's a convex quadratic, the optimum is unique — that's *why* there
is exactly one stable answer.

## What we found when we ran it
- Our implementation vs a naive "nearest labelled point" baseline, same 400 points,
  same 2 revealed labels, ten fixed seeds: **0.45% error vs 22.36%**. We won all ten runs.
- Loosen the wires and it collapses: σ = 0.05/0.10/0.15/0.30/0.60 →
  **0.15% / 0.10% / 0.45% / 14.07% / 32.59%**. At 0.60 we **lose 8 of the 10** runs.
  So the advantage is not the algorithm, it's the wire scale.
- α cost: reaching the answer took 1,405 iterations at α=0.99 and 10,860 at α=0.999.
  Bumping α buys diffusion, paid for in time.
- Our own checks: iterating and solving the linear system agree to ~1e-10 with
  **zero label disagreements**; the loop stops when the *residual* is tiny, which means
  the true error is ~100× larger than that residual at α=0.99 (exactly `tol/(1−α)`, as the
  theory predicts); and one identity the paper prints in Section 2 is **false as matrix
  algebra but harmless for classification** (a missing `D` scaling, which cancels in argmax).

## The catch (this is the 30%-weighted part: critique)
Choosing σ needs a labelled test set. The whole point of the method is that you don't
have one. So it is excellent when the scale is known, and **untestable when it isn't** —
which is precisely the regime it's sold for. The paper admits this and reports every
method at its own best setting. We fixed σ=0.15 *before* running so we weren't cheating,
which is honest but arbitrary — and our sweep shows what "arbitrary" costs.

## Ours vs the paper's evidence
The paper ran real data: 3,874 USPS digits (1–4, 16×16) and 3,970 newsgroup documents,
averaged over 100 trials, α=0.99, with 1-NN, RBF-SVM, and a harmonic/CMN competitor. On
digits, consistency wins clearly; on text the harmonic method is better at some label
budgets, so the paper is **not** claiming universal superiority — and its Figure 5 has no
error bars. Ours is a toy: newly generated moons, one label per class, scored only on the
points whose labels were hidden, and it can only claim the *mechanism*, not their numbers.
That is also the answer to "inductive?": we are **transductive** — we may look at the test
points' coordinates, never their labels. Unseen points were never tested, by us or the
paper's Figure 4 idea (which we didn't implement).

## "What did your group actually contribute?" — the 15-second answer
Re-derived the convergence and the regularised-objective view; implemented the algorithm
from scratch in NumPy; verified every theoretical claim numerically (which is how we found
the false-but-harmless identity); ran a leakage-free ten-seed comparison with a bandwidth
sweep; and written up where the method's advantage depends on a parameter you can't
validate.

## Numbers to have in your mouth without looking
`0.45% vs 22.36%` · `32.59% and 8/10 lost at σ=0.60` · `α=0.99` · `2 labels, 400 points` ·
`zero disagreements with the direct solve, ~1e-10` · `μ=(1−α)/α` · `eigenvalues in [−1,1]`.
