"""Learning with Local and Global Consistency (Zhou et al., Section 2)."""

import numpy as np


def squared_distances(X):
    """Return the squared Euclidean distance between each pair of rows."""
    X = np.asarray(X, dtype=float)
    differences = X[:, None, :] - X[None, :, :]
    return np.sum(differences ** 2, axis=-1)


def affinity_matrix(X, sigma):
    """Build the Gaussian affinity matrix with no self-edges."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    distances_sq = squared_distances(X)
    W = np.exp(-0.5 * (distances_sq / sigma) / sigma)
    np.fill_diagonal(W, 0.0)
    return W


def fit_scores(X, observed_labels, *, sigma, alpha, tol=1e-12, max_iter=20_000):
    """Return (scores, classes, iterations) from the fixed-point iteration.

    Rows of X are points. Labels are integers, with -1 marking unknown labels.
    Score columns correspond to the sorted observed class labels. Stop when
    the largest absolute change in any score falls below tol.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(observed_labels)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError("X must contain at least two points with finite features")
    if labels.shape != (len(X),) or labels.dtype.kind not in "iu" or np.any(labels < -1):
        raise ValueError("observed_labels must contain one integer per point, each >= -1")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    # Y is one-hot for labeled points and zero for unlabeled points.
    labeled = labels != -1
    classes, columns = np.unique(labels[labeled], return_inverse=True)
    if classes.size == 0:
        raise ValueError("at least one point must have an observed label")
    Y = np.zeros((len(X), len(classes)))
    Y[labeled, columns] = 1.0

    # Form S = D^(-1/2) W D^(-1/2) by scaling both axes of W.
    W = affinity_matrix(X, sigma)
    degrees = W.sum(axis=1)
    if np.any(degrees <= 0):
        raise ValueError("a point has zero total affinity; try a larger sigma")
    inverse_sqrt_degree = 1.0 / np.sqrt(degrees)
    S = inverse_sqrt_degree[:, None] * W * inverse_sqrt_degree[None, :]

    # Every point receives neighbor scores and the original label signal.
    scores = Y.copy()
    label_signal = (1.0 - alpha) * Y
    for iteration in range(1, max_iter + 1):
        next_scores = alpha * (S @ scores) + label_signal
        if np.max(np.abs(next_scores - scores)) < tol:
            return next_scores, classes, iteration
        scores = next_scores
    raise RuntimeError(f"LGC did not converge within {max_iter} iterations")


def predict(X, observed_labels, *, sigma, alpha):
    """Return one predicted class per point for the group's experiment runner."""
    scores, classes, _ = fit_scores(X, observed_labels, sigma=sigma, alpha=alpha)
    return classes[np.argmax(scores, axis=1)]
