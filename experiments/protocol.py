"""Small, deterministic two-moons protocol implemented directly with NumPy.

The generator follows the usual two-semicircle geometry; it does not reproduce
the original paper's undisclosed toy sample or reuse scikit-learn source code.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Trial:
    X: np.ndarray
    truth: np.ndarray
    observed: np.ndarray  # -1 means unrevealed, 0/1 are the two class labels

    @property
    def test_mask(self) -> np.ndarray:
        return self.observed == -1


def make_trial(seed: int, n_samples: int = 400, noise: float = 0.08) -> Trial:
    """Generate fresh data and sample exactly one revealed point per class.

    Separate child RNG streams isolate label sampling from noise generation.
    Class membership is used to enforce the stated sampling protocol, then
    unrevealed labels are removed before a predictor receives the trial.
    """
    if seed < 0 or n_samples < 6 or n_samples % 2:
        raise ValueError("Use a nonnegative seed and an even n_samples >= 6.")
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative.")
    data_seed, label_seed = np.random.SeedSequence(seed).spawn(2)
    data_rng = np.random.default_rng(data_seed)
    label_rng = np.random.default_rng(label_seed)
    theta = np.linspace(0.0, np.pi, n_samples // 2)
    upper = np.column_stack((np.cos(theta), np.sin(theta)))
    lower = np.column_stack((1.0 - np.cos(theta), 0.5 - np.sin(theta)))
    X = np.vstack((upper, lower))
    X += data_rng.normal(0.0, noise, size=X.shape)
    truth = np.repeat([0, 1], n_samples // 2)
    permutation = data_rng.permutation(n_samples)
    X, truth = X[permutation], truth[permutation]
    observed = np.full(n_samples, -1, dtype=int)
    for label in (0, 1):
        index = label_rng.choice(np.flatnonzero(truth == label))
        observed[index] = label
    return Trial(X=X, truth=truth, observed=observed)


def nearest_neighbor(X: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Euclidean 1-NN using only revealed examples; ties choose first index."""
    labeled = np.flatnonzero(observed >= 0)
    if not len(labeled):
        raise ValueError("1-NN requires at least one revealed label.")
    squared_distances = ((X[:, None, :] - X[labeled][None, :, :]) ** 2).sum(axis=2)
    return observed[labeled[np.argmin(squared_distances, axis=1)]]


def checked_predictions(predictions: np.ndarray, n_samples: int) -> np.ndarray:
    """Require one finite 0/1 class prediction per point, not score matrices."""
    predictions = np.asarray(predictions)
    if predictions.shape != (n_samples,) or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Predictor must return shape (n_samples,) with labels 0 or 1.")
    return predictions.astype(int)


def evaluate(trial: Trial, predictions: np.ndarray) -> dict:
    predictions = checked_predictions(predictions, len(trial.X))
    mask = trial.test_mask
    if not mask.any():
        raise ValueError("Evaluation requires unrevealed examples.")
    errors = int(np.count_nonzero(predictions[mask] != trial.truth[mask]))
    n_test = int(mask.sum())
    return {
        "n_labeled": int((~mask).sum()),
        "n_test": n_test,
        "errors": errors,
        "error_rate": errors / n_test,
        "accuracy": 1.0 - errors / n_test,
    }


def predict_lgc(predictor: Callable, trial: Trial, sigma: float, alpha: float) -> np.ndarray:
    """Adapter boundary: the core receives coordinates and masked labels only.

    Copies prevent an implementation from mutating the shared evaluation data.
    See experiments/README.md for the proposed callable interface.
    """
    predictions = predictor(trial.X.copy(), trial.observed.copy(), sigma=sigma, alpha=alpha)
    return checked_predictions(predictions, len(trial.X))
