#!/usr/bin/env python3
"""Role B checks: does the label-propagation iteration agree with the
linear-system solution, and do the paper's Section 2/3 equations hold up?

Paper: Zhou, Bousquet, Lal, Weston, Scholkopf. "Learning with Local and Global
Consistency." NIPS 2003 (our Group 28 paper).

Section 2 defines the iteration
    F(t+1) = alpha * S @ F(t) + (1 - alpha) * Y,    alpha in (0, 1)
and claims (Eq. 2-3) it converges to the closed form, and Section 3 (Eq. 4-6)
shows the same F* is the minimiser of a regularised objective Q(F), after the
substitution alpha = 1/(1+mu). So "iterate until fixed point" and "solve the
linear system (I - alpha*S) F = (1 - alpha) Y" are two names for the same
object.

Four blocks, all runnable against either our own toy or Role C's protocol:

  [convergence] iteration vs direct linear solve, spectral radius of alpha*S,
                error decay slope vs the predicted ln(alpha), and the paper's
                claim that the limit does not depend on F(0)
  [variants]    the operator equivalences claimed in the last paragraph of
                Section 2 and in Section 4's experiment setup:
                  (I-aS)^-1 Y   vs  (I-aP)^-1 Y   vs  (I-aP')^-1 Y
                  (D-aW)^-1 D Y  vs  (D-aW)^-1 Y
                (P = D^-1 W, row-stochastic. We check the algebraic identities
                literally, then how much the argmax classification differs.)
  [optimality]  Q(F) from Eq. 4: stationarity ||dQ/dF|| at F*, and that random
                perturbations of F* all increase Q (convex quadratic minimum)
  [--protocol]  the same convergence check on the datasets from
                experiments/protocol.py (Role C's generator), seeds 0-9, so
                numbers line up with results/baseline/. Also reports what the
                propagation reference would score per seed (pending A's core;
                NOT a team result until cross-checked against src/)

Run (from the repository root):
    python3 -m experiments.convergence_check --sweep 0.5 0.9 0.95 0.99 0.999
    python3 -m experiments.convergence_check --protocol
    python3 -m experiments.convergence_check --sweep 0.5 0.9 0.99 --plot
      (the plot needs matplotlib, not otherwise required:
       `uv run --with matplotlib -m experiments.convergence_check --plot ...`)

Writes: results/convergence_check.csv (or --out path; --protocol defaults to
results/protocol_check.csv) and, with --plot, results/convergence_rate.png.

Deps: numpy (developed on 2.3.5), Python 3.12+; matplotlib only for --plot.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_moons(n: int, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-circles, same pattern as the paper's Figure 1(a).
    Construction follows sklearn's make_moons (MIT licence), reimplemented here
    so the repo needs no extra dependency. Returns X (n, 2), y (n,) in {0, 1}."""
    half = n // 2
    t_out = np.linspace(0.0, math.pi, n - half)
    t_in = np.linspace(0.0, math.pi, half)
    outer = np.column_stack([np.cos(t_out), np.sin(t_out)])
    inner = np.column_stack([1.0 - np.cos(t_in), 1.0 - np.sin(t_in) - 0.5])
    X = np.vstack([outer, inner]) + noise * rng.standard_normal((n, 2))
    y = np.concatenate([np.zeros(n - half, dtype=int), np.ones(half, dtype=int)])
    return X, y


def build_graph(X: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paper Section 2, steps 1-2. Returns (W, d, S) with W the zero-diagonal
    RBF affinity, d the degree vector (D_ii), S = D^(-1/2) W D^(-1/2)."""
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1)
    W = np.exp(-d2 / (2.0 * sigma**2))
    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1)
    if np.any(d == 0.0):
        raise ValueError("isolated point: row sum of W is zero, raise sigma")
    dinv_sqrt = 1.0 / np.sqrt(d)
    return W, d, dinv_sqrt[:, None] * W * dinv_sqrt[None, :]


def build_Y(n: int, y: np.ndarray, labelled: tuple[int, int]) -> np.ndarray:
    """Clamping matrix: Y_ij = 1 if x_i is labelled class j, else 0.
    One revealed label per class, like the paper's Figure 1(a)."""
    Y = np.zeros((n, 2))
    Y[labelled[0], y[labelled[0]]] = 1.0
    Y[labelled[1], y[labelled[1]]] = 1.0
    return Y


def closed_form(S: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Direct linear-system solution: (I - alpha S) F = (1 - alpha) Y."""
    n = Y.shape[0]
    return np.linalg.solve(np.eye(n) - alpha * S, (1.0 - alpha) * Y)


def iterate(S: np.ndarray, Y: np.ndarray, alpha: float, reference: np.ndarray,
            F0: np.ndarray, tol: float, max_iter: int) -> tuple[np.ndarray, int, list[tuple[int, float]]]:
    """The paper's iteration, tracking max-abs error against the closed-form
    reference every checkpoint. Stops on the *observable* residual criterion
    ||F(t) - F(t-1)||_inf < tol, which is what a user of the algorithm has."""
    beta = 1.0 - alpha
    F = F0.copy()
    curve: list[tuple[int, float]] = []
    residual = math.inf
    t = 0
    for t in range(1, max_iter + 1):
        F_next = alpha * (S @ F) + beta * Y
        residual = np.abs(F_next - F).max()
        F = F_next
        if t % 10 == 0 or residual < tol:
            curve.append((t, float(np.abs(F - reference).max())))
        if residual < tol:
            break
    return F, t, curve


def slope_of_log_error(curve: list[tuple[int, float]]) -> float | None:
    """Least-squares slope of ln(error) vs t over the part of the curve where
    the error is in [1e-9, 1e-2] (transient done, float noise not yet reached)."""
    pts = [(t, e) for t, e in curve if 1e-9 <= e <= 1e-2]
    if len(pts) < 8:
        return None
    t = np.array([p[0] for p in pts], dtype=float)
    e = np.log(np.array([p[1] for p in pts]))
    return float(np.polyfit(t, e, 1)[0])


def variants_audit(W: np.ndarray, d: np.ndarray, S: np.ndarray, Y: np.ndarray,
                   alpha: float) -> list[dict]:
    """Last paragraph of Section 2 + Section 4's variant list, checked literally.

    Section 2 claims:  (I - aP')^-1 Y  is equivalent to  (D - aW)^-1 D Y
    Section 4 lists variant (2) as:     (D - aW)^-1 Y   (no D factor on Y)
    Algebra says:      (I - aP')^-1 Y = D (D - aW)^-1 Y, which is *not* the
    same thing unless D commutes with (D-aW)^-1. We print the measured gaps
    instead of trusting any derivation, including ours.
    """
    n = len(d)
    Dm = np.diag(d)
    I = np.eye(n)
    P = W / d[:, None]  # D^-1 W, row-stochastic
    main = np.linalg.solve(I - alpha * S, Y)          # (I - aS)^-1 Y, Eq. 3
    v_p = np.linalg.solve(I - alpha * P, Y)           # Section 2 variant 1
    v_pt = np.linalg.solve(I - alpha * P.T, Y)        # Section 2 variant "P'"
    v_dw_dy = np.linalg.solve(Dm - alpha * W, Dm @ Y)   # Section 2's written form
    v_dw_y = np.linalg.solve(Dm - alpha * W, Y)         # Section 4's written form
    v_d_dw_y = Dm @ v_dw_y                              # algebra-corrected P' form

    def gap(a: np.ndarray, b: np.ndarray) -> float:
        scale = max(np.abs(a).max(), np.abs(b).max(), 1e-30)
        return float(np.abs(a - b).max() / scale)

    def mismatch(a: np.ndarray, b: np.ndarray) -> int:
        return int((a.argmax(axis=1) != b.argmax(axis=1)).sum())

    rows = [
        {"claim": "sec2: (I-aP)^-1 Y == (D-aW)^-1 D Y",
         "rel_gap": gap(v_p, v_dw_dy), "argmax_diff_vs_main": mismatch(v_p, main)},
        {"claim": "sec2: (I-aP')^-1 Y == (D-aW)^-1 D Y  (as written in paper)",
         "rel_gap": gap(v_pt, v_dw_dy), "argmax_diff_vs_main": mismatch(v_pt, main)},
        {"claim": "algebra: (I-aP')^-1 Y == D (D-aW)^-1 Y (corrected form)",
         "rel_gap": gap(v_pt, v_d_dw_y), "argmax_diff_vs_main": mismatch(v_pt, main)},
        {"claim": "sec4 variant(2) as written: (D-aW)^-1 Y vs (I-aP')^-1 Y",
         "rel_gap": gap(v_dw_y, v_pt), "argmax_diff_vs_main": mismatch(v_dw_y, main)},
        {"claim": "main S vs variant (I-aP)^-1 Y (row-scaling)",
         "rel_gap": gap(main, v_p), "argmax_diff_vs_main": mismatch(main, v_p)},
    ]
    return rows


def q_functional(W: np.ndarray, d: np.ndarray, Y: np.ndarray, F: np.ndarray, mu: float) -> float:
    """Eq. 4 exactly: smoothness term sum_ij W_ij ||F_i/sqrt(d_i) - F_j/sqrt(d_j)||^2
    halved, plus mu * sum_i ||F_i - Y_i||^2."""
    Fhat = F / np.sqrt(d)[:, None]
    diff = Fhat[:, None, :] - Fhat[None, :, :]
    smooth = 0.5 * float((W[:, None] * (diff ** 2).sum(axis=2)).sum())
    fit = mu * float(((F - Y) ** 2).sum())
    return smooth + fit


def optimality_check(S: np.ndarray, W: np.ndarray, d: np.ndarray, Y: np.ndarray,
                     alpha: float, rng: np.random.Generator) -> dict:
    """Section 3: F* = argmin Q at mu = (1-alpha)/alpha. Test the stationarity
    equation F* - S F* + mu (F* - Y) = 0 and that random perturbations raise Q."""
    mu = (1.0 - alpha) / alpha
    Fstar = closed_form(S, Y, alpha)
    grad = Fstar - S @ Fstar + mu * (Fstar - Y)
    q0 = q_functional(W, d, Y, Fstar, mu)
    worse = 0
    scales = [0.05, 0.2, 1.0]
    trials = 3 * len(scales) * (20 // (3 * len(scales)))  # keeps the printed denominator honest
    for i in range(trials):
        eps = scales[i % len(scales)]
        pert = Fstar + eps * rng.standard_normal(Fstar.shape)
        if q_functional(W, d, Y, pert, mu) > q0:
            worse += 1
    return {
        "mu_from_alpha": round(mu, 6),
        "max_abs_stationarity_residual": f"{np.abs(grad).max():.3e}",
        "Q_at_Fstar": f"{q0:.6e}",
        "perturbations_raising_Q": f"{worse}/{trials}",
    }


def run_alpha(S: np.ndarray, Y: np.ndarray, y_true: np.ndarray, labelled: tuple[int, int],
              alpha: float, tol: float, max_iter: int, rng: np.random.Generator) -> tuple[dict, list[tuple[int, float]]]:
    eig_abs = np.abs(np.linalg.eigvalsh(S)).max()  # S symmetric, real spectrum
    F_ref = closed_form(S, Y, alpha)
    F_iter, iters, curve = iterate(S, Y, alpha, F_ref, Y, tol, max_iter)

    F_zero, _, _ = iterate(S, Y, alpha, F_ref, np.zeros_like(Y), tol, max_iter)
    F_rand, _, _ = iterate(S, Y, alpha, F_ref, rng.random(Y.shape), tol, max_iter)
    indep = max(np.abs(F_zero - F_iter).max(), np.abs(F_rand - F_iter).max())

    pred_slope = math.log(alpha)
    emp_slope = slope_of_log_error(curve)
    pred_iters = math.ceil(math.log(tol) / pred_slope)
    unlab = np.ones(Y.shape[0], dtype=bool)
    unlab[list(labelled)] = False
    acc = float((F_iter.argmax(axis=1)[unlab] == y_true[unlab]).mean())
    mismatch = int((F_iter.argmax(axis=1) != F_ref.argmax(axis=1)).sum())

    row = {
        "alpha": alpha,
        "max_abs_eig_S": round(float(eig_abs), 6),
        "predicted_iterations": pred_iters,
        "iterations_to_tol": iters,
        "max_abs_diff_vs_solve": f"{np.abs(F_iter - F_ref).max():.3e}",
        "predicted_slope_ln_alpha": round(pred_slope, 5),
        "empirical_slope_log_err_vs_t": round(emp_slope, 5) if emp_slope is not None else "n/a",
        "initialisation_independence": f"{indep:.3e}",
        "label_mismatches_vs_solve": mismatch,
        "unlabelled_accuracy": round(acc, 4),
    }
    return row, curve


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")


def plot_rate(curves: list[tuple[float, list[tuple[int, float]]]], path: Path, tol: float,
              caption: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for alpha, curve in curves:
        t = [c[0] for c in curve]
        e = [c[1] for c in curve]
        ax.semilogy(t, e, label=f"alpha = {alpha}")
    alpha0, curve0 = curves[0]
    t0 = [c[0] for c in curve0]
    ax.semilogy(t0, [curve0[0][1] * alpha0 ** t for t in t0], ls="--", c="gray",
                label="reference: C·α₀ᵗ (predicted slope)")
    ax.axhline(tol, c="k", lw=0.6, ls=":")
    ax.set_xlabel("iteration t")
    ax.set_ylabel("max-abs error vs closed form")
    ax.set_title("Label propagation decays geometrically (log error, straight = αᵗ)")
    ax.legend(fontsize=8)
    fig.text(0.5, 0.015, caption or "Our results on generated two-moons data.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=180)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def protocol_mode(args, rng) -> None:
    """Same convergence check on Role C's generator (experiments/protocol.py).
    Uses his sigma/alpha defaults so the accuracy column is directly comparable
    with results/baseline/ summary.csv."""
    import importlib
    protocol = importlib.import_module("experiments.protocol")
    rows = []
    for seed in range(args.protocol_seeds):
        trial = protocol.make_trial(seed, args.n_protocol, args.noise_protocol)
        observed = trial.observed
        labelled_idx = np.flatnonzero(observed >= 0)
        W, d, S = build_graph(trial.X, args.sigma)
        Y = np.zeros((len(trial.X), 2))
        for i in labelled_idx:
            Y[i, int(observed[i])] = 1.0
        row, _ = run_alpha(S, Y, trial.truth, tuple(labelled_idx), args.alpha,
                           args.tol, args.max_iter, rng)
        row |= {"seed": seed, "sigma": args.sigma, "n": args.n_protocol,
                "noise": args.noise_protocol, "n_labelled": len(labelled_idx)}
        rows.append(row)
        print(f"seed {seed}: iters {row['iterations_to_tol']}, diff vs solve "
              f"{row['max_abs_diff_vs_solve']}, prop acc {row['unlabelled_accuracy']:.4f} "
              f"(mismatch {row['label_mismatches_vs_solve']})")
    diffs = [float(r["max_abs_diff_vs_solve"]) for r in rows]
    accs = [r["unlabelled_accuracy"] for r in rows]
    print(f"\nprotocol summary: all {len(rows)} seeds converge to the solve within "
          f"{max(diffs):.3e}; mean propagation accuracy {np.mean(accs):.4f} "
          f"(C's 1-NN baseline: mean error 0.2236, results/baseline/summary.csv)")
    print("propagation rows are B's reference check pending A's core in src/; "
          "not a team comparison yet")
    out = args.out or REPO_ROOT / "results" / "protocol_check.csv"
    write_csv(rows, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200, help="data points (default 200)")
    ap.add_argument("--sigma", type=float, default=None,
                    help="RBF bandwidth; defaults to 0.1 standalone, 0.15 in --protocol "
                         "(C's proposed LGC config; the paper leaves the toy one unstated)")
    ap.add_argument("--noise", type=float, default=0.1, help="moons position noise")
    ap.add_argument("--alpha", type=float, default=0.99, help="paper uses alpha = 0.99")
    ap.add_argument("--sweep", type=float, nargs="*", default=None, metavar="A",
                    help="run one convergence row per alpha instead of a single run")
    ap.add_argument("--tol", type=float, default=1e-12, help="stop when residual ||F(t)-F(t-1)||inf < tol")
    ap.add_argument("--max-iter", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true", help="write results/convergence_rate.png (needs matplotlib)")
    ap.add_argument("--protocol", action="store_true",
                    help="run on experiments/protocol.py data (Role C's generator) instead")
    ap.add_argument("--protocol-seeds", type=int, default=10)
    ap.add_argument("--n-protocol", type=int, default=400, help="n_samples for C's make_trial")
    ap.add_argument("--noise-protocol", type=float, default=0.08, help="noise for C's make_trial")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.sigma is None:
        args.sigma = 0.15 if args.protocol else 0.1
    rng = np.random.default_rng(args.seed)
    print(f"run header: numpy {np.__version__}, seed {args.seed}, tol {args.tol}, sigma {args.sigma}")

    if args.protocol:
        protocol_mode(args, rng)
        return

    X, y_true = make_moons(args.n, args.noise, rng)
    labelled = (args.n // 2 - 1, args.n - 1)  # one revealed point per class
    W, d, S = build_graph(X, args.sigma)
    Y = build_Y(args.n, y_true, labelled)
    print(f"standalone toy: n={args.n} two-moons, sigma={args.sigma}, noise={args.noise}, "
          f"labelled={labelled}")

    alphas = args.sweep if args.sweep else [args.alpha]
    rows, curves = [], []
    for a in alphas:
        row, curve = run_alpha(S, Y, y_true, labelled, a, args.tol, args.max_iter, rng)
        rows.append(row)
        curves.append((a, curve))
        print()
        for key, val in row.items():
            print(f"  {key:32s} {val}")

    print("\n[variants audit] relative gaps (0 = identical to float precision):")
    vrows = variants_audit(W, d, S, Y, args.alpha)
    for r in vrows:
        print(f"  {r['claim']:62s} gap {r['rel_gap']:.3e}  argmax-diff-vs-main {r['argmax_diff_vs_main']}")

    print("\n[optimality check] Section 3 Eq. 4-6 at alpha =", args.alpha)
    opt = optimality_check(S, W, d, Y, args.alpha, rng)
    for key, val in opt.items():
        print(f"  {key:32s} {val}")

    args.out = args.out or REPO_ROOT / "results" / "convergence_check.csv"
    write_csv(rows, args.out)
    write_csv(vrows, args.out.parent / "variants_audit.csv")
    if args.plot:
        plot_rate(curves, args.out.parent / "convergence_rate.png", args.tol,
                  caption=f"Our results on generated two-moons data, n = {args.n}, "
                          f"sigma = {args.sigma}, noise = {args.noise}, seed = {args.seed}.")


if __name__ == "__main__":
    main()
