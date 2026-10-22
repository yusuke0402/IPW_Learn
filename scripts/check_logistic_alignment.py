"""Check IPW logistic selection, convergence, bootstrap, and CLI settings.

uv run scripts/check_logistic_alignment.py
"""

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logistic_weight as logistic
import main
from scripts import nonlinear_gamma_ipw as experiment
from weight import compute_weights, ipw_estimate

REFERENCE_ROOTS = []


class LogisticAlignmentChecks(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        np.random.seed(7)
        self.tx, self.ty, self.sx, self.sy = experiment.generate_data("data_scenario_1", 100, .5, 50)
        self.grid = [.01, 1., 100.]

    def test_default_settings_and_reference_weights_match(self):
        np.testing.assert_array_equal(logistic.DEFAULT_C_GRID, np.logspace(-4, 4, 17))
        self.assertEqual(logistic.DEFAULT_CV_FOLDS, 5)
        self.assertEqual(logistic.DEFAULT_CV_SEED, 2024)
        self.assertEqual(logistic.DEFAULT_SOLVER, "newton-cholesky")
        self.assertEqual(logistic.DEFAULT_TOL, 1e-8)
        fit = logistic.fit_membership_logistic(self.tx[:, 1:], self.sx[:, 1:])
        info = fit.weight_selection_
        self.assertEqual(info["max_iter"], 1000)
        self.assertEqual(info["probability_clip"], 1e-12)
        for root in REFERENCE_ROOTS:
            spec = importlib.util.spec_from_file_location("reference_logistic_" + root.name, root / "logistic_weight.py")
            reference = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reference)
            other = reference.fit_membership_logistic(self.tx[:, 1:], self.sx[:, 1:])
            self.assertEqual(info, other.weight_selection_)
            np.testing.assert_array_equal(logistic.source_odds(fit, self.sx[:, 1:]),
                                          reference.source_odds(other, self.sx[:, 1:]))

    def test_cv_scores_match_independent_fits(self):
        tx, sx = self.tx[:, 1:], self.sx[:, 1:]
        fit = logistic.fit_membership_logistic(tx, sx, C_grid=self.grid)
        x = np.vstack([tx, sx])
        labels = np.r_[np.ones(len(tx)), np.zeros(len(sx))]
        scores = []
        for strength in self.grid:
            losses = []
            for train, valid in StratifiedKFold(5, shuffle=True, random_state=2024).split(x, labels):
                reference = LogisticRegression(C=strength, solver="lbfgs", tol=1e-10, max_iter=2000)
                reference.fit(x[train], labels[train])
                losses.append(log_loss(labels[valid], reference.predict_proba(x[valid])))
            scores.append(np.mean(losses))
        np.testing.assert_allclose(fit.weight_selection_["cv_log_loss"], scores, atol=2e-7, rtol=0)
        self.assertEqual(fit.weight_selection_["C"], self.grid[np.argmin(scores)])

    def test_weights_use_group_covariates_and_preserve_random_state(self):
        before = np.random.get_state()
        first = ipw_estimate(self.tx, self.ty, self.sx, self.sy, method="propensity",
                             min_ess=0, propensity_C_grid=self.grid)
        second = ipw_estimate(self.tx, self.ty * 100, self.sx, self.sy - 200, method="propensity",
                              min_ess=0, propensity_C_grid=self.grid)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        np.testing.assert_array_equal(first["source_weights"], second["source_weights"])
        selected = first["propensity_selection"]
        fit = logistic.fit_membership_logistic(self.tx[:, 1:], self.sx[:, 1:], C_grid=self.grid)
        np.testing.assert_array_equal(first["source_weights"], logistic.source_odds(fit, self.sx[:, 1:]))
        self.assertEqual(first["ess_diagnostics"]["propensity_C"], selected["C"])

    def test_fixed_c_validation_fold_reduction_and_nonconvergence(self):
        fit = logistic.fit_membership_logistic(self.tx[:, 1:], self.sx[:, 1:], cv=False, C=2.5)
        self.assertEqual(fit.weight_selection_["C"], 2.5)
        self.assertFalse(fit.weight_selection_["cv"])
        fit = logistic.fit_membership_logistic(self.tx[:3, 1:], self.sx[:4, 1:], C_grid=[1.])
        self.assertEqual(fit.weight_selection_["cv_folds"], 3)
        for tx, grid in ((self.tx[:1, 1:], [1.]), (self.tx[:, 1:], [0., 1.])):
            with self.assertRaises(ValueError):
                logistic.fit_membership_logistic(tx, self.sx[:, 1:], C_grid=grid)
        with patch.object(logistic.LogisticRegressionCV, "fit", side_effect=ConvergenceWarning("test")):
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                compute_weights("propensity", self.tx, self.sx)

    def test_main_and_nonlinear_reselect_c_for_every_bootstrap(self):
        config = yaml.safe_load((ROOT / "configs/config.yaml").read_text())
        config["hyperparameters"].update(n_features=2, min_ess=0, propensity_C_grid=self.grid)
        config["dataset"].update(target_number=24, source_number=35)
        for entry in ("main", "nonlinear"):
            records = []

            def capture(tx, sx, **kwargs):
                fit = logistic.fit_membership_logistic(tx, sx, **kwargs)
                records.append(fit.weight_selection_)
                return fit

            with patch("propensityscore.fit_membership_logistic", side_effect=capture):
                if entry == "main":
                    row = main._run_one_trial(3, config, "propensity", 2, main._weight_kwargs(config))
                    diagnostics = json.loads(row["ess_diagnostics"])
                    bootstrap = json.loads(row["boot_ess_diagnostics"])
                else:
                    row = experiment.run_one_trial(3, "data_scenario_1", 35, .5, 2, 24,
                                                    "propensity", dict(min_ess=0, propensity_C_grid=self.grid))
                    diagnostics = json.loads(row["ipw_ess_diagnostics"])
                    bootstrap = json.loads(row["ipw_boot_ess_diagnostics"])
            self.assertEqual(len(records), 3)
            self.assertTrue(all(record["cv"] and record["C_grid"] == self.grid for record in records))
            self.assertEqual(diagnostics["propensity_C"], records[0]["C"])
            self.assertEqual([record["propensity_C"] for record in bootstrap],
                             [record["C"] for record in records[1:]])

    def test_cli_uses_configured_fixed_c_and_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(tmp)
            config = yaml.safe_load((ROOT / "configs/config.yaml").read_text())
            config["hyperparameters"].update(propensity_cv=False, propensity_C=2.5,
                                             propensity_penalty="l2", min_ess=0)
            path = folder / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            common = ["experiment", "--config", str(path), "--weight-method", "propensity",
                      "--scenarios", "data_scenario_1", "--source-ns", "35", "--target-n", "24",
                      "--gammas", "0.5", "--trials", "1", "--bootstraps", "2", "--n-jobs", "1"]
            cases = [([], False, 2.5), (["--propensity-C", ".75"], False, .75),
                     (["--propensity-cv", "--propensity-C-grid", ".01", "1"], True, None)]
            for index, (extra, cv, strength) in enumerate(cases):
                out = folder / str(index)
                with patch.object(sys, "argv", common + ["--out-dir", str(out)] + extra):
                    experiment.main()
                meta = json.loads((out / "run_meta.json").read_text())
                self.assertEqual(meta["propensity"]["propensity_cv"], cv)
                if strength is not None:
                    self.assertEqual(meta["propensity"]["propensity_C"], strength)
                else:
                    self.assertEqual(meta["propensity"]["propensity_C_grid"], [.01, 1.])
            saved = (folder / "0" / "run_meta.json").read_bytes()
            with patch.object(sys, "argv", common + ["--out-dir", str(folder / "0"), "--propensity-C", ".75"]):
                with self.assertRaisesRegex(ValueError, "new output directory"):
                    experiment.main()
            self.assertEqual((folder / "0" / "run_meta.json").read_bytes(), saved)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iwl-root", type=Path)
    parser.add_argument("--aipw-root", type=Path)
    args, rest = parser.parse_known_args()
    REFERENCE_ROOTS = [path for path in (args.iwl_root, args.aipw_root) if path]
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)
