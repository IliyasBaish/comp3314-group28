# B segment — rehearsal cue card (2 min, speaker: Dylan)

Improvise from the beats. Nothing here is a sentence to read aloud.
Numbers are taken from the merged CSVs, so they are safe to quote cold.
Fixed 22 Nov: slide B1 now prints `F* = (1−α)(I − αS)⁻¹Y` — keep saying the `(1−α)`.
If the theory feels foreign before you drill: read `docs/b-crash-briefing.md` first
(10 minutes, plain language, same numbers), then come back to this card.


## 1. Beats (90–120 s, you have ~30 s of slack — spend it on numbers, not padding)

| Time | Beat | Cue words (say these, build the sentence yourself) |
|---|---|---|
| 0:00–0:15 | Handoff from A | "A showed the loop; my job is why the loop can't fail" · normalisation is the quiet hero |
| 0:15–0:50 | Convergence | similar to `D⁻¹W` → eigenvalues of a stochastic matrix · symmetric → all real, all in **[−1,1]**, top eigenvalue exactly 1 (`‖S‖₂ = 1`, we measured it) · so `αS` is a **contraction** · Neumann series → closed form `F* = (1−α)(I−αS)⁻¹Y` · reached **from any start** · PageRank damping / power iteration is the same trick |
| 0:50–1:05 | Global consistency made explicit | `(I−αS)⁻¹ = Σ(αS)^k` = **diffusion kernel** · `α^k` weights path length k · a label travels multiple hops, not just neighbours |
| 1:05–1:40 | Third view: it's an optimiser | `Q(F)` = smoothness (graph-Laplacian energy in degree-normalised coords) + clamping/fidelity term · `∂Q = 0` → same fixed point · `μ = (1−α)/α` · **convex quadratic ⇒ unique** · course hook: ridge regression / normal equations, spectral clustering uses the *same* `D^-1/2WD^-1/2`, manifold regularisation is the general frame |
| 1:40–2:00 | We checked it, not just derived it | iteration vs `np.linalg.solve` agrees ~**1e-10** (worst 8×10⁻¹¹ over C's ten seeds), zero argmax mismatches · stopping error = **`tol/(1−α)`** · accuracy **0.77 → 1.00** as α goes 0.5 → 0.999 · "iterate = solve = minimise: three views of one object" |

## 2. Numbers in your back pocket

- α sweep iters: **29 / 167 / 326 / 1,405 / 10,860** (α = 0.5 / 0.9 / 0.95 / 0.99 / 0.999); worst-case bounds `ln tol / ln α` = 40 / 263 / 539 / 2,750 / 27,618 → bound holds, is loose (the λ=1 mode nearly cancels; measured slope −0.0132 vs ln α = −0.0101).
- `tol = 1e-12`, α = 0.99 → predicted error 1e-10, observed **9.1e-11**; α = 0.999 → 1e-9 vs **7.5e-10**.
- `Q` stationarity residual **2.1e-17**; 18/18 random perturbations raised Q.
- Team headline (C's, don't present it as yours): 0.45% ± 0.54 vs 1-NN 22.36% ± 4.22, ten fixed seeds, σ=0.15, α=0.99.
- Toy convergence run is 200 points, σ=0.1, noise 0.1 — *not* the same config as the 400-point comparison.

## 3. Do NOT say (rails)

- `F* = (I−αS)⁻¹Y` without the `(1−α)`. If pressed: the paper's Eq. 3 omits the constant; it scales every row equally, so the argmax is unchanged.
- "μ = α" — it is `μ = (1−α)/α`, `α = 1/(1+μ)`.
- "it converges because α < 1" alone. The reason is the **spectrum bound** (`|λ(S)| ≤ 1`) — that's the trap answer, and TA/prof can push on it.
- "we replicated the paper's results." Ours is a **qualitative** check on newly generated toy data; no paper error rate is re-derived.
- "LGC wins on every paired trial" unscoped — true only at σ ≤ 0.30. At σ=0.60: 32.59% and 1-NN wins 8/10.
- "σ=0.15 is the paper's toy setting." It's **ours**, declared before the run; the paper's 0.15 is a *text* bandwidth in a different feature space.
- "all consistency variants perform equally" — the paper shows one variant doing substantially worse.
- The §2 transpose identity point: say **"false as matrix algebra, harmless for classification"** (gap 0.819; corrected form with the extra D verifies to 1.4e-15; ≤6/200 predictions shift). Don't claim the algorithm is broken.
- Don't attribute a sentence to the paper unless you can point at it on the page. Safe: "the symmetric normalisation is the same one spectral clustering uses (Ng–Jordan–Weiss), which the paper cites" — that's *your* observation backed by their reference list. Unsafe: "the paper says steps 1–2 are identical to NGJ" (I could not re-verify that wording in the PDF tonight — don't quote it).
- Anything about inductive / out-of-sample behaviour — we didn't implement the Figure-4 extension.
- Don't quote error bars for the paper's Figure 5; it has none. And don't call our SD a confidence interval.

## 4. Q&A bank for B

1. **Why zero diagonal in W?** No self-reinforcement; a point must not vote for itself.
2. **Disconnected graph?** `D⁻¹W` then has multiple λ=1 modes; labels can't cross components — propagation is per-component.
3. **What does α control?** How far labels diffuse before damping stops them. α→1 ≈ harmonic/global, α→0 ≈ clamped/local. Ours: accuracy 0.77→1.00, cost 1,405→10,860 iterations.
4. **Why symmetric S and not row-stochastic P?** Symmetry is what makes the spectrum real; the two differ, and the paper itself tests both variants (the Eq. 6 derivation is for S).
5. **Is `I−αS` always invertible?** For α<1 yes: eigenvalues of `I−αS` lie in [1−α, 1+α], bounded away from 0.
6. **What would you change to make this scale?** Sparse k-NN graph + iterative solve (CG); dense W is O(n²) — at n=1e5 that's 1e10 entries ≈ 80 GB float64.
7. **Is your 0.45% a general claim?** No — generated two moons, one label per class, unrevealed points only, fixed bandwidth chosen before the run. It tests the mechanism, and the bandwidth sweep shows how fragile the advantage is.
8. **What did the bonus actually check?** That the iteration and the direct solve produce the same F (and the same labels), and that the observable residual stops (1−α) earlier than the true error does.

## 5. Cold-explain protocol (do this in rehearsal — the team is shaky, and Q&A is 3 min examinable by anyone)

Rule: 45 s, no notes, no slides, one person, one claim. If you can't land it, that's a Monday risk — mark it and re-run tomorrow.

| # | Claim to explain cold | Ground truth for self-marking |
|---|---|---|
| 1 | What problem does this solve that 1-NN can't? | Labels are scarce, examples aren't; local rule can't cross the gap between crescents, graph diffusion can |
| 2 | What are the two consistency assumptions and how do they differ? | Local = neighbours share labels (smoothness along short edges); global = same cluster/manifold chain shares labels; the method enforces both in one energy |
| 3 | Where does σ enter, and why is it the model, not a footnote? | Only through W: it sets which edges exist; our sweep 0.10%→32.59% across five bandwidths |
| 4 | Why is `‖S‖₂ = 1` important? | It bounds the spectrum so `αS` contracts; that gives the geometric rate and the Neumann closed form |
| 5 | Why is the fixed point unique? | Q is a strictly convex quadratic in F (μ>0 clamping term) — stationarity is also sufficient |
| 6 | What is transductive evaluation, and what does it forbid? | Test points' *features* may be used in the graph; their *labels* may not. No unseen-vector claim |
| 7 | What did each of us actually verify? | A: predictor + 0/400 label diffs vs B's closed form on all ten seeds · B: convergence, tol/(1−α), Q optimality, identity audit · C: fixed protocol, leakage-free 10-seed comparison + baselines · D: bandwidth analysis (run through C's harness), deck assembly, intro + critique in the report |
| 8 | Name one limitation you'd fix first | Label-scarce model selection — no honest way to pick σ with two labels; we fixed it before the run and paid for it in the sweep |
| 9 | What does `α = 0.99` in the paper buy, and cost? | Diffusion reach (accuracy on our toy) at the cost of iterations; ~2,750 predicted vs 1,405 measured at 0.99 |
| 10 | Where does the paper's own evidence stay qualified? | Text results: harmonic/CMN method beats consistency at some label budgets; no error bars; authors flag parameter selection under label scarcity |

## 6. Also settle tonight (all four are online once more before Monday)

- Everyone reads main's report sections — including Thumula's two, drafted 21 Nov — and posts "OK" on issue #4 by 26–27 Nov.
- 29 Nov: Iliyas submits the report named `Group_28.pdf`; Dylan verifies upload + receipt. The slides file has the *same* name locally — report copy lives in `submission/Group_28.pdf`.
- Nobody has a record that the 15 Nov slides upload was confirmed on CourseNet — check it tonight.
- Stale PR #6 (beamer deck, superseded by Canva) — close or leave, their call.
- Monday is double-booked (this talk + the COMP3330 demo): agree the order, and pre-record the 3330 demo as backup.
