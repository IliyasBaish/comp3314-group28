#!/usr/bin/env python3
"""Role B bonus check: does the label-propagation iteration agree with the
linear-system solution?

Paper: Zhou, Bousquet, Lal, Weston, Scholkopf. "Learning with Local and Global
Consistency." NIPS 2003 (our Group 28 paper).

Section 2 defines the iteration
    F(t+1) = alpha * S @ F(t) + (1 - alpha) * Y,    alpha in (0, 1)
and claims (Eq. 2-3) it converges to F* = (I - alpha*S)^-1 Y, independent of the
initial value. Section 3 (Eq. 4-6) shows the same F* is the unique minimiser of
the regularised objective Q(F) (smoothness + fitting) after the substitution
alpha = 1/(1+mu). So "iterate until fixed point" and "solve the linear system
(I - alpha*S) F = (1 - alpha) Y" are two names for the same object. This script
checks that numerically.

Three claims tested, on the paper's own toy pattern (two moons, Section 4.1):
  1. agreement     : converged iteration == direct linear solve (max-abs diff)
  2. contraction   : eigenvalues of S lie in [-1, 1] (S is similar to the
                     row-stochastic D^-1 W), so the error decays like alpha^t.
                     We fit log(error) vs t and compare to the predicted slope
                     ln(alpha), and compare iterations-to-tolerance against
                     the prediction ln(tol)/ln(alpha).
  3. F(0)-free     : the limit is the same for F(0) = Y, F(0) = 0, random F(0).

Self-contained numpy (no sklearn) so it does not block on the core
implementation in src/. Everything is seeded and printed in the run header, and
S is built exactly as the paper prescribes (zero-diagonal RBF affinity,
symmetric normalisation). Only the bandwidth sigma and the two revealed labels
are our choices, same setup the paper shows in Figure 1(a)/2.

Run:
    python3 experiments/convergence_check.py
    python3 experiments/convergence_check.py --alpha 0.99 --n 400 --sigma 0.15
    python3 experiments/convergence_check.py --sweep 0.5 0.9 0.99 0.999

Writes: results/convergence_check.csv (one row per alpha).

Deps: numpy (developed on 2.3.5), Python 3.12.
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


def build_S(X: np.ndarray, sigma: float) -> np.ndarray:
    """Paper Section 2, steps 1-2: W_ij = exp(-||xi - xj||^2 / 2 sigma^2) with
    W_ii = 0, then S = D^(-1/2) W D^(-1/2)."""
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1)
    W = np.exp(-d2 / (2.0 * sigma**2))
    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1)
    if np.any(d == 0.0):
        raise ValueError("isolated point: row sum of W is zero, raise sigma")
    dinv_sqrt = 1.0 / np.sqrt(d)
    return dinv_sqrt[:, None] * W * dinv_sqrt[None, :]


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


def run_alpha(S: np.ndarray, Y: np.ndarray, y_true: np.ndarray, labelled: tuple[int, int],
              alpha: float, tol: float, max_iter: int, rng: np.random.Generator) -> dict:
    eig_abs = np.abs(np.linalg.eigvalsh(S)).max()  # S symmetric, real spectrum
    F_ref = closed_form(S, Y, alpha)
    F_iter, iters, curve = iterate(S, Y, alpha, F_ref, Y, tol, max_iter)

    # claim 3: limit independent of the starting point
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

    return {
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200, help="data points (default 200)")
    ap.add_argument("--sigma", type=float, default=0.1, help="RBF bandwidth (paper leaves the toy one unstated)")
    ap.add_argument("--noise", type=float, default=0.1, help="moons position noise")
    ap.add_argument("--alpha", type=float, default=0.99, help="paper uses alpha = 0.99")
    ap.add_argument("--sweep", type=float, nargs="*", default=None, metavar="A",
                    help="run one row per alpha instead of a single run")
    ap.add_argument("--tol", type=float, default=1e-12, help="stop when residual ||F(t)-F(t-1)||inf < tol")
    ap.add_argument("--max-iter", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "convergence_check.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X, y_true = make_moons(args.n, args.noise, rng)
    labelled = (args.n // 2 - 1, args.n - 1)  # one revealed point per class
    S = build_S(X, args.sigma)
    Y = build_Y(args.n, y_true, labelled)

    header = {
        "script": Path(__file__).name, "n": args.n, "sigma": args.sigma, "noise": args.noise,
        "seed": args.seed, "tol": args.tol, "numpy": np.__version__,
        "labelled_points": labelled,
    }
    print("run header:", header)

    alphas = args.sweep if args.sweep else [args.alpha]
    rows = [run_alpha(S, Y, y_true, labelled, a, args.tol, args.max_iter, rng) for a in alphas]

    for row in rows:
        print()
        for key, val in row.items():
            print(f"  {key:32s} {val}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
