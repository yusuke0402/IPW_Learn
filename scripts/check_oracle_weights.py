"""Check exact Gaussian weights, reference-estimator integration, and resume safety.

uv run scripts/check_oracle_weights.py
"""

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from scipy.stats import multivariate_normal
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main
from data import DataSets
from experiment_diagnostics import InsufficientESSError
from experiment_execution import run_weighted_bootstrap
from oracle_weight import _gaussian_logpdf, gaussian_oracle_metadata, oracle_density_ratio
from scripts import nonlinear_gamma_ipw as experiment
from weight import compute_weights, ipw_estimate, weighted_mean_difference


class OracleWeightChecks(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        self.config = yaml.safe_load((ROOT / "configs/config.yaml").read_text())
        self.means = dict(target=np.array([1., -1.]), source=np.array([0., 2.]))
        self.covs = dict(target=np.diag([2., 3.]), source=np.diag([1., 4.]))
        self.x = np.array([[0., 0.], [1., -1.], [0., 2.], [2., 1.]])

    def reference(self, x, means, covs):
        return np.exp(multivariate_normal.logpdf(x, mean=means["target"], cov=covs["target"])
                      - multivariate_normal.logpdf(x, mean=means["source"], cov=covs["source"]))

    def test_direction_and_raw_scale_match_diagonal_hand_formula(self):
        vt, vs = np.diag(self.covs["target"]), np.diag(self.covs["source"])
        log_ratio = (.5 * np.log(vs / vt).sum()
                     - .5 * (((self.x - self.means["target"]) ** 2 / vt).sum(axis=1)
                              - ((self.x - self.means["source"]) ** 2 / vs).sum(axis=1)))
        weights = oracle_density_ratio(self.x, self.means, self.covs)
        np.testing.assert_allclose(weights, np.exp(log_ratio), rtol=2e-15)
        reversed_means = dict(target=self.means["source"], source=self.means["target"])
        reversed_covs = dict(target=self.covs["source"], source=self.covs["target"])
        np.testing.assert_allclose(weights * oracle_density_ratio(self.x, reversed_means, reversed_covs), 1.)
        equal = dict(target=np.zeros(2), source=np.zeros(2))
        equal_covs = dict(target=np.eye(2), source=np.eye(2))
        tx, sx = np.c_[np.ones(2), self.x[:2]], np.c_[np.ones(4), self.x]
        _, ws = compute_weights("oracle", tx, sx, means=equal, covariances=equal_covs)
        np.testing.assert_array_equal(ws, np.ones(4))

    def test_all_configured_2d_10d_scenarios_match_scipy_and_exclude_intercept(self):
        rng = np.random.default_rng(7)
        for dimension, scenarios in ((2, range(1, 5)), (10, range(1, 4))):
            for scenario in scenarios:
                with self.subTest(dimension=dimension, scenario=scenario):
                    config = copy.deepcopy(self.config)
                    config["hyperparameters"]["n_features"] = dimension
                    config["scenario"]["data_scenario_id"] = f"data_scenario_{scenario}"
                    data = DataSets(config)
                    x = rng.multivariate_normal(data.means["source"], data.covariances["source"], 12)
                    expected = self.reference(x, data.means, data.covariances)
                    np.testing.assert_allclose(oracle_density_ratio(x, data.means, data.covariances), expected,
                                               rtol=2e-12, atol=1e-14)
                    _, actual = compute_weights("oracle", np.c_[np.ones(4), x[:4]], np.c_[np.ones(12), x],
                                                 means=data.means, covariances=data.covariances)
                    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-14)
                    with self.assertRaises(ValueError):
                        oracle_density_ratio(np.c_[np.ones(12), x], data.means, data.covariances)

    def test_invalid_gaussian_inputs_fail_instead_of_silently_using_lower_triangle(self):
        invalid_covs = [np.array([[1., 999.], [0., 1.]]), np.zeros((2, 2)),
                        np.diag([1., -1.]), np.eye(3), np.full((2, 2), np.nan),
                        np.array([[1., 0.], [0., np.inf]]), {"not": "an array"}]
        for cov in invalid_covs:
            with self.subTest(cov=cov), self.assertRaises(ValueError):
                oracle_density_ratio(self.x, self.means, dict(target=cov, source=np.eye(2)))
        for x, mean in ((np.array([[np.nan, 0.]]), np.zeros(2)),
                        (np.array([[np.inf, 0.]]), np.zeros(2)),
                        (self.x, np.array([np.nan, 0.])), (self.x, np.zeros(3)),
                        (self.x.ravel(), np.zeros(2))):
            with self.subTest(x=x, mean=mean), self.assertRaises(ValueError):
                _gaussian_logpdf(x, mean, np.eye(2))
        for means, covs in ((None, self.covs), ({"target": np.zeros(2)}, self.covs),
                            (self.means, {"source": np.eye(2)})):
            with self.assertRaises(ValueError):
                oracle_density_ratio(self.x, means, covs)

    def test_underflow_is_zero_and_overflow_is_failure_without_clipping(self):
        x = np.full((4, 1), 40.)
        covs = dict(target=np.eye(1), source=np.eye(1))
        means = dict(target=np.array([0.]), source=np.array([40.]))
        np.testing.assert_array_equal(oracle_density_ratio(x, means, covs), np.zeros(4))
        tx, sx = np.c_[np.ones(4), np.zeros(4)], np.c_[np.ones(4), x]
        with self.assertRaises(InsufficientESSError):
            ipw_estimate(tx, np.ones(4), sx, np.zeros(4), method="oracle", min_ess=0,
                          means=means, covariances=covs)
        with self.assertRaises(ValueError):
            ipw_estimate(tx, np.ones(4), sx, np.zeros(4), method="oracle", min_ess=0,
                          bootstrap=True, means=means, covariances=covs)
        reverse = dict(target=means["source"], source=means["target"])
        with self.assertRaisesRegex(ValueError, "not representable"):
            oracle_density_ratio(x, reverse, covs)
        run = run_weighted_bootstrap(
            lambda a, b, c, d: ipw_estimate(a, b, c, d, method="oracle", means=reverse, covariances=covs),
            tx, np.ones(4), sx, np.zeros(4), 2, 5)
        self.assertEqual(run["point_error_type"], "numerical")
        self.assertEqual(len(run["boot_errors"]), 2)
        self.assertEqual(run["boot_n"], 0)

    def test_hajek_mean_is_stable_without_changing_raw_weights_or_normal_case(self):
        wt, ws = np.array([.25, 1., 3., 4.]), np.array([1., .5, 7., 2.])
        ty, sy = np.array([1., -2., 4., 5.]), np.array([3., 1., -1., 2.])
        expected = np.sum(wt * ty) / np.sum(wt) - np.sum(ws * sy) / np.sum(ws)
        self.assertAlmostEqual(weighted_mean_difference(wt, ty, ws, sy, min_ess=0), expected, places=14)
        raw = np.full(4, 1e308)
        before = raw.copy()
        self.assertEqual(weighted_mean_difference(raw, np.ones(4), raw, np.full(4, 2.), min_ess=0), -1.)
        np.testing.assert_array_equal(raw, before)
        center = np.sqrt(2 * np.log(1e308))
        tx = np.c_[np.ones(4), np.zeros(4)]
        sx = np.c_[np.ones(4), np.full(4, center)]
        means = dict(target=np.array([center]), source=np.array([0.]))
        covs = dict(target=np.eye(1), source=np.eye(1))
        result = ipw_estimate(tx, np.ones(4), sx, np.full(4, 2.), method="oracle", min_ess=0,
                              means=means, covariances=covs)
        self.assertEqual(result["estimate"], -1.)
        self.assertEqual(result["ess"], 4.)
        np.testing.assert_allclose(result["source_weights"], 1e308, rtol=2e-13)
        self.assertGreater(result["max_weight"], 9.9e307)

    def test_main_2d_10d_and_nonlinear_bootstrap_keep_true_distributions_fixed(self):
        for dimension, scenario, entry in ((2, 4, "main"), (10, 2, "main"), (2, 4, "nonlinear")):
            config = copy.deepcopy(self.config)
            config["hyperparameters"].update(n_features=dimension, min_ess=0)
            config["dataset"].update(target_number=24, source_number=35)
            config["scenario"]["data_scenario_id"] = f"data_scenario_{scenario}"
            seen = []

            def capture(x, means, covs):
                spec = gaussian_oracle_metadata(means, covs)
                seen.append(spec)
                actual = oracle_density_ratio(x, means, covs)
                np.testing.assert_allclose(actual, self.reference(x, means, covs), rtol=2e-12, atol=1e-14)
                return actual

            with self.subTest(dimension=dimension, entry=entry), patch("weight.oracle_density_ratio", side_effect=capture):
                if entry == "main":
                    row = main._run_one_trial(3, config, "oracle", 2, main._weight_kwargs(config))
                    diagnostics = json.loads(row["ess_diagnostics"])
                else:
                    row = experiment.run_one_trial(3, "data_scenario_4", 35, .5, 2, 24,
                                                    "oracle", dict(min_ess=0))
                    diagnostics = json.loads(row["ipw_ess_diagnostics"])
                self.assertEqual(len(seen), 3)
                self.assertTrue(all(spec == seen[0] for spec in seen))
                self.assertEqual(diagnostics["oracle_distribution"], seen[0])

    def test_cli_records_true_distributions_and_refuses_runtime_truth_change(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp) / "oracle"
            argv = ["experiment", "--weight-method", "oracle", "--scenarios", "data_scenario_4",
                    "--trials", "2", "--bootstraps", "2", "--source-ns", "35", "--target-n", "24",
                    "--gammas", "0.5", "--n-jobs", "1", "--min-ess", "0", "--out-dir", str(out)]
            with patch.object(sys, "argv", argv):
                experiment.main()
            path = out / "run_meta.json"
            saved = json.loads(path.read_text())
            self.assertEqual(set(saved["oracle_distributions"]), set(experiment.MEANS))
            sc4 = saved["oracle_distributions"]["data_scenario_4"]
            self.assertEqual(sc4["means"], {"target": [0., 3.], "source": [2., 0.]})
            self.assertEqual(sc4["ratio"], "p_target/p_source")
            self.assertEqual(saved["resume_contract"]["oracle_distributions"], saved["oracle_distributions"])
            with patch.object(sys, "argv", argv):
                experiment.main()
            previous = path.read_bytes()
            changed = copy.deepcopy(experiment.MEANS["data_scenario_4"])
            changed["target"][0] = .125
            with patch.dict(experiment.MEANS, data_scenario_4=changed), patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "new output directory"):
                    experiment.main()
            self.assertEqual(path.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main(verbosity=2)
