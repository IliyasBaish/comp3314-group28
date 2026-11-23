# B's actual slots — read-aloud script + line-by-line annotations
(From the run-of-show you screenshotted: slide 5 = 1:10, slide 7 = 0:30, slide 10 = 0:25. You have THREE appearances, 2:05 total.)

---

## SLIDE 5 — "Mathematical foundation and course connections" (1:10)

### SCRIPT (read it as written; ~175 words)

Why does this loop have an answer at all?

Everything comes from that **S** on the slide — W normalised at both ends. Two things follow. Its eigenvalues all sit between minus one and one; we measured it, the top one is exactly one. And alpha is less than one, so every pass through the loop shrinks what's left. That's a contraction — it *has* to settle.

And when it settles, we can write it down instead of iterating. That's the equation in the middle: F-star equals one-minus-alpha, times the inverse of I minus alpha-S, times Y. The contraction is exactly why that inverse exists. And that inverse is a diffusion kernel: expand it and alpha-to-the-k weights paths of length k. That is global consistency written as algebra.

One connection to this course: it's ridge regression in graph clothing. Penalise roughness along the edges, keep fidelity to the labels you have, set the derivative to zero — same equation. And the D-to-the-minus-half normalisation is the same trick spectral clustering uses.

So: iterate, solve, minimise. Three names for one object.

### ANNOTATION — what each sentence means and where it can be poked

| You say | What it actually means | If someone pushes |
|---|---|---|
| "Why does this loop have an answer at all?" | The iteration is not a heuristic; it's guaranteed to converge | — |
| "S = W normalised at both ends" | `S = D^(-1/2) W D^(-1/2)`; D is the row-sum degree matrix, so a busy point can't dominate | "Why normalise?" → so the spectrum behaves; the raw W doesn't |
| "eigenvalues between −1 and 1, top exactly 1" | S is a rescaled copy of a "who-do-I-tell" matrix → same eigenvalues; symmetric → all real. ‖S‖₂ = 1.0 in our runs | "Any catch?" → if the graph is disconnected you get *several* eigenvalues at 1, and labels can't cross components |
| "alpha < 1, so every pass shrinks what's left — a contraction" | Multiplying by α squeezes the error geometrically; that's the whole convergence proof | This is the trap question: never say "it converges because α<1" alone — the reason is the eigenvalue bound **plus** α<1 |
| "F* = (1−α)(I−αS)⁻¹Y" | The closed form. `(1−α)` out front scales every row equally, so it never changes which class wins | "The paper writes it without (1−α)" → right, constant, argmax unchanged |
| "the contraction is why that inverse exists" | Eigenvalues of `I−αS` lie in [1−α, 1+α], never 0 → invertible | say exactly that if pressed |
| "that inverse is a diffusion kernel; α^k weights paths of length k" | Expand `(I−αS)⁻¹ = I + αS + α²S² + …` — the k-th term is label flow along k-step paths | This is your best "understanding" line. It's why a label reaches the far end of a crescent |
| "ridge regression in graph clothing" | The objective is smoothness + μ·fidelity; derivative zero gives that linear system. μ = (1−α)/α | "So what's the penalty?" → graph roughness (the Laplacian), not weight size. **Never say μ = α** |
| "same normalisation as spectral clustering" | Ng–Jordan–Weiss normalise the affinity the same way. Say it as *your* observation, not as a quote from the paper | — |
| "iterate, solve, minimise" | All three are the same F*. That's the punchline of the theory | — |

---

## SLIDE 7 — "Implementation verification" (0:30, ~70 words)

### SCRIPT

We didn't just trust the algebra. Running the loop and solving that linear system directly give the same answer — to about one part in ten billion — and identical labels in all ten of our trials. One honest wrinkle: the loop stops when the *change* from one step to the next gets small, and that change underreports the true error by a factor of one over one-minus-alpha. At alpha nought point nine nine, that's a hundred times. The theory predicted its own stopping error — we measured it, and it lands.

### ANNOTATION
- "same answer" = iteration vs `np.linalg.solve` on `(I−αS)F=(1−α)Y`; agreement ~1e-10, worst 8×10⁻¹¹ across C's ten seeds; **zero** label mismatches.
- "change from one step to the next" = the residual `‖F(t+1) − F(t)‖`, the only thing you can observe while running.
- "underreports by 1/(1−α)" = residual carries an extra (1−α) factor, so stopping at residual 1e-12 means the real error is ~1e-10 at α=0.99, ~1e-9 at α=0.999. Predicted and measured matched (9.1e-11 and 7.5e-10).
- If asked "so is your loop unreliable?" → No: it's *predictable* — we know exactly how much error to expect from a chosen tolerance.
- Keep the "paper's Section 2 identity" point OUT of this slot unless asked; if asked: "false as matrix algebra — a missing D — harmless for the labels, because it scales rows equally."

---

## SLIDE 10 — "Conclusion" (0:25, ~60 words)

### SCRIPT

Conclusion. Normalise the graph and the diffusion becomes a contraction with a closed form — and that closed form is the unique minimum of a smoothness-plus-fidelity energy. We re-derived it, verified it numerically, and reproduced the two-moons effect: nought point four five percent error against twenty-two for one-NN. Loosen the bandwidth and it stops working. So the derivation is not the hard part. Choosing the scale is — and that's exactly what you can't validate when labels are scarce.

### ANNOTATION
- "unique minimum" — the objective is a convex quadratic with the fidelity term making it strictly convex, so one optimum.
- "0.45% vs 22.36%" is σ=0.15, ten fixed seeds, one label per class, toy data. Say "our toy two-moons run" so nobody thinks it's a real-world number.
- "loosen the bandwidth" — at σ=0.60 we measured 32.59% and 1-NN wins eight of ten. If asked, quote that; it *is* the critique.
- "can't validate" — the closing point of the whole deck; it's also D's slide 9 theme, so hand back to him if he's still standing.

---

## If you blank, these three sentences are enough
1. "The normalisation puts S's eigenvalues inside minus one to one, so with alpha less than one the loop is a contraction — it must settle."
2. "When it settles, F star is one-minus-alpha times the inverse of I minus alpha-S, times Y, and that inverse exists precisely because it's a contraction."
3. "That inverse is a diffusion kernel — alpha-to-the-k weights paths of length k — and the same F star is the minimum of a smoothness-plus-fidelity energy, which is ridge regression with a graph."

## Do NOT say (quick rails)
- `F* = (I−αS)⁻¹Y` without the `(1−α)` out front.
- "μ = α" → it's `μ = (1−α)/α`.
- "it converges because α < 1" with no mention of the eigenvalue bound.
- "we replicated the paper" — toy data, mechanism only. Also: σ=0.15 is *ours*, chosen before the run.
- "LGC wins every trial" — only at σ ≤ 0.30. At σ=0.60 we lose 8/10.
- Don't call our SD across seeds a confidence interval; don't quote error bars for the paper's Figure 5 (it has none).

## Slide-render note
Your screenshot shows the F* equation split across two boxes on slide 5, so it may look clipped on the projector. If it renders badly, say: "the `(1−α)` out front scales every row equally, so it doesn't change the classification — the important part is the inverse" and keep moving. Don't stop to fix slides live.
