"""Check default grids and original/bootstrap weight screening independently.

uv run scripts/check_ipw_bootstrap_alignment.py
"""

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_diagnostics import InsufficientESSError, InsufficientPositiveWeightsError
from experiment_execution import diagnostic_columns, run_weighted_bootstrap, summarize_trials, variant_columns
from experiment_io import write_run_meta
from result import Result
from weight import ULSIF, compute_weights, ipw_estimate

REFERENCE_ROOTS = []


class BootstrapAlignmentChecks(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1., 0., 0.], [1., 1., 2.], [1., 2., 4.], [1., 3., 6.]])
        self.y = np.arange(4, dtype=float)

    def estimate(self, weights, *, bootstrap=False, minimum=3, dimension=2, method="ulsif", x=None):
        x = np.ones((len(weights), dimension + 1)) if x is None else x
        y = np.arange(len(weights), dtype=float)
        with patch("weight.compute_weights", return_value=(np.ones(len(x)), np.asarray(weights))):
            return ipw_estimate(x, y, x, y, method=method, min_ess=minimum, bootstrap=bootstrap)

    def test_default_grid_matches_references_without_overriding_them(self):
        expected = {
            "SIGMA_FACTORS": (.3, .5, .7, 1., 1.4, 2., 3.),
            "LAMBDA_GRID": (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.),
            "CV_FOLDS": 5, "CV_SEED": 12345, "MAX_BASIS": 100,
        }
        for name, value in expected.items():
            self.assertEqual(getattr(ULSIF, name), value)
        for reference, filename, classname in REFERENCE_ROOTS:
            tree = ast.parse((reference / filename).read_text())
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == classname)
            assignments = {node.targets[0].id: node.value for node in cls.body
                           if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
            for name, value in expected.items():
                with self.subTest(reference=reference.name, name=name):
                    self.assertEqual(ast.literal_eval(assignments["RATIO_" + name]), value)

    def test_bootstrap_records_low_ess_but_original_requires_ess(self):
        for method in ("uniform", "propensity", "ulsif", "oracle"):
            with self.subTest(method=method):
                weights = [1000., 1., 1., 0.]
                with self.assertRaises(InsufficientESSError):
                    self.estimate(weights, method=method)
                result = self.estimate(weights, bootstrap=True, minimum=100, method=method)
                diagnostics = result["ess_diagnostics"]
                self.assertLess(result["ess"], 3)
                self.assertEqual(diagnostics["full"]["positive_weights"], 3)
                self.assertFalse(diagnostics["ess_threshold_applied"])
                self.assertEqual(diagnostics["fit_context"], "bootstrap")
                self.assertEqual(diagnostics["min_positive_weights"], 3)
                np.testing.assert_array_equal(result["source_weights"], weights)

    def test_positive_row_boundary_in_2d_and_10d_and_duplicates_count_as_rows(self):
        for dimension in (2, 10):
            p = dimension + 1
            weights = np.r_[np.ones(p - 1), np.zeros(2)]
            self.estimate(weights, minimum=0, dimension=dimension)
            with self.assertRaises(InsufficientPositiveWeightsError) as caught:
                self.estimate(weights, minimum=0, bootstrap=True, dimension=dimension)
            self.assertEqual(caught.exception.ess_diagnostics["min_positive_weights"], p)
            # Every design row is the same: resampled copies still count as rows for this gate.
            weights = np.r_[1000., np.ones(p - 1), 0.]
            result = self.estimate(weights, minimum=100, bootstrap=True, dimension=dimension)
            self.assertEqual(result["ess_diagnostics"]["full"]["positive_weights"], p)

    def test_zero_density_bootstrap_is_not_rescued_and_original_fallback_is_preserved(self):
        tx = self.x.copy()
        sx = self.x.copy()
        sx[:, 1:] += 1e6  # Source kernels underflow to zero against all target basis centers.
        for minimum in (0, 3):
            for bootstrap in (False, True):
                diagnostics = {}
                with patch.object(ULSIF, "_select_hyperparams", return_value=(1., .1)):
                    _, weights = compute_weights("ulsif", tx, sx, min_ess=minimum,
                                                 bootstrap=bootstrap, diagnostics=diagnostics)
                rescued = minimum == 0 and not bootstrap
                np.testing.assert_array_equal(weights, np.ones(4) if rescued else np.zeros(4))
                self.assertEqual(diagnostics["raw"]["positive_weights"], 0)
                self.assertEqual(diagnostics["zero_weight_fallback"], rescued)
                if bootstrap:
                    with self.assertRaises(InsufficientPositiveWeightsError):
                        self.estimate(weights, bootstrap=True, minimum=minimum)

    def test_bootstrap_failures_are_counted_as_positive_weights_and_keep_ci_cutoff(self):
        for failed in (1, 2):
            calls = iter([np.ones(4)] + [np.array([1., 1., 0., 0.])] * failed
                         + [np.array([1000., 1., 1., 0.])] * (20 - failed))

            def fit(a, b, c, d, bootstrap=False):
                return self.estimate(next(calls), bootstrap=bootstrap)

            run = run_weighted_bootstrap(fit, self.x, self.y, self.x, self.y, 20, 4,
                bootstrap_estimator=lambda *args: fit(*args, bootstrap=True))
            self.assertEqual(run["boot_n"], 20 - failed)
            self.assertEqual({error["type"] for error in run["boot_errors"]}, {"positive_weights"})
            row = {**diagnostic_columns(run, "ipw_"), **variant_columns(run, "ipw")}
            self.assertEqual(row["ipw_boot_failed_ESS"], 0)
            self.assertEqual(row["ipw_boot_failed_positive_weights"], failed)
            stats = summarize_trials(pd.DataFrame([row]), "ipw", 0.)
            self.assertEqual(stats["bootstrap_failed_positive_weights"], failed)
            self.assertEqual(stats["bootstrap_failed_ESS"], 0)
            self.assertEqual(run["intervals"]["estimate"]["error"],
                             "" if failed == 1 else "bootstrap_failure_rate_exceeded")

    def test_saved_rules_and_resume_contract_record_bootstrap_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            meta = dict(method="IPW", min_ess=3, max_bootstrap_failure_rate=.05,
                        target_n=4, bootstraps=2, n_trials=1)
            write_run_meta(out, ROOT, **meta)
            path = out / "run_meta.json"
            saved = json.loads(path.read_text())
            self.assertEqual(saved["ess_scope"], "original_sample_only")
            self.assertFalse(saved["bootstrap_ess_threshold_applied"])
            self.assertEqual(saved["bootstrap_min_positive_weights"], 3)
            self.assertEqual(saved["resume_contract"]["ess_contract_version"], 2)
            saved["resume_contract"]["ess_contract_version"] = 1
            path.write_text(json.dumps(saved))
            previous = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "new output directory"):
                write_run_meta(out, ROOT, **meta)
            self.assertEqual(path.read_bytes(), previous)

        import main
        for dimension in (2, 10):
            config = yaml.safe_load((ROOT / "configs/config.yaml").read_text())
            config["hyperparameters"].update(n_features=dimension, n_trial=1, min_ess=3)
            config["scenario"]["data_scenario_id"] = "data_scenario_1"
            config["dataset"].update(target_number=20, source_number=20)
            row = main._run_one_trial(0, config, "uniform", 2, main._weight_kwargs(config))
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                config["output_dir"] = tmp
                Result.save_results([row], config)
                saved = yaml.safe_load(next(Path(tmp).glob("summary*.yaml")).read_text())
                rule = saved["feasibility_rule"]
                self.assertFalse(rule["bootstrap_ess_threshold_applied"])
                self.assertEqual(rule["bootstrap_min_positive_weights"], dimension + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iwl-root", type=Path)
    parser.add_argument("--aipw-root", type=Path)
    args, rest = parser.parse_known_args()
    if args.iwl_root:
        REFERENCE_ROOTS.append((args.iwl_root, "importance_weighted_learn.py", "ImportaceWeightLearn"))
    if args.aipw_root:
        REFERENCE_ROOTS.append((args.aipw_root, "density_ratio.py", "ULSIFDensityRatio"))
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)
