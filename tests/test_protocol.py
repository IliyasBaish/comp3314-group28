"""Protect evaluation fairness and reproducibility, not an LGC implementation."""

import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.protocol import Trial, checked_predictions, evaluate, make_trial, nearest_neighbor, predict_lgc
from experiments.run import main


class ProtocolTests(unittest.TestCase):
    def test_repeatable_trial_and_exact_label_budget(self):
        first, second = make_trial(3), make_trial(3)
        np.testing.assert_array_equal(first.X, second.X)
        np.testing.assert_array_equal(first.observed, second.observed)
        self.assertEqual(np.count_nonzero(first.observed == 0), 1)
        self.assertEqual(np.count_nonzero(first.observed == 1), 1)
        self.assertEqual(first.test_mask.sum(), 398)
        self.assertFalse(np.array_equal(first.X, make_trial(4).X))

    def test_evaluation_excludes_revealed_points(self):
        trial = Trial(np.zeros((4, 2)), np.array([0, 1, 0, 1]), np.array([0, 1, -1, -1]))
        # Deliberately wrong on both revealed examples and one hidden example.
        result = evaluate(trial, np.array([1, 0, 0, 0]))
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["n_test"], 2)
        self.assertEqual(result["error_rate"], 0.5)

    def test_predictor_cannot_see_truth_or_mutate_shared_inputs(self):
        trial = make_trial(0)
        original_X, original_observed = trial.X.copy(), trial.observed.copy()

        def spy(X, observed, *, sigma, alpha):
            self.assertEqual(np.count_nonzero(observed == -1), 398)
            self.assertEqual((sigma, alpha), (0.15, 0.99))
            X[:] = 0
            observed[:] = 0
            return np.zeros(len(X), dtype=int)

        predict_lgc(spy, trial, sigma=0.15, alpha=0.99)
        np.testing.assert_array_equal(trial.X, original_X)
        np.testing.assert_array_equal(trial.observed, original_observed)

    def test_nearest_neighbor_uses_revealed_points_only(self):
        X = np.array([[0.0, 0], [10.0, 0], [1.0, 0], [9.0, 0]])
        np.testing.assert_array_equal(nearest_neighbor(X, np.array([0, 1, -1, -1])), [0, 1, 0, 1])

    def test_invalid_predictions_fail_instead_of_becoming_results(self):
        for bad in (np.zeros((4, 2)), np.array([0, 1, -1, 0]), np.array([0, 1, np.nan, 0])):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                checked_predictions(bad, 4)

    def test_invalid_data_configuration(self):
        for kwargs in ({"seed": -1}, {"seed": 0, "n_samples": 5}, {"seed": 0, "noise": np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_trial(**kwargs)

    def test_baseline_run_is_reproducible_and_explicitly_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = ["--seeds", "0", "1", "--n-samples", "20", "--out", directory]
            with redirect_stdout(io.StringIO()):
                main(command)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            with redirect_stdout(io.StringIO()):
                main(command)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "baseline only; LGC pending")
            with (root / "trials.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual({r["n_test"] for r in rows}, {"18"})
            self.assertFalse((root / "paired.csv").exists())

    def test_paired_runner_reuses_trials_for_every_method(self):
        # This test-only adapter is 1-NN, not an LGC implementation or result.
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(io.StringIO()):
                main(["--seeds", "0", "1", "--n-samples", "20", "--sigmas", "0.1", "0.2",
                      "--predictor", "tests.test_protocol:test_only_adapter", "--out", directory])
            with (Path(directory) / "paired.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(float(row["lgc_minus_1nn_error"]) == 0 for row in rows))

    def test_bad_adapter_does_not_create_result_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "should-not-exist"
            with self.assertRaises(ValueError):
                main(["--seeds", "0", "1", "--n-samples", "20", "--predictor",
                      "tests.test_protocol:bad_adapter", "--out", str(output)])
            self.assertFalse(output.exists())


def test_only_adapter(X, observed, *, sigma, alpha):
    return nearest_neighbor(X, observed)


def bad_adapter(X, observed, *, sigma, alpha):
    return np.full(len(X), -1)


if __name__ == "__main__":
    unittest.main()
