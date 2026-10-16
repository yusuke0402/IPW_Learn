import argparse
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io
import json
import numpy as np
import pandas as pd
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1]
)
args = parser.parse_args()
sys.path.insert(0, str(args.root))
from experiment_diagnostics import InsufficientESSError, weight_diagnostics
from experiment_execution import (
    run_weighted_bootstrap,
    diagnostic_columns,
    variant_columns,
    summarize_trials,
)
from experiment_io import write_run_meta, load_or_run_trials

AIPW = args.root.name == "AIPW-Learn"
if AIPW:
    import aipw_learn as model
    from density_ratio import ULSIFDensityRatio
    from scripts import nonlinear_gamma_aipw as experiment
else:
    import weight as model
    from scripts import nonlinear_gamma_ipw as experiment
import main
from result import Result


class Checks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(31)
        self.x = np.column_stack([np.ones(20), rng.normal(size=(20, 2))])
        self.y = self.x @ [2, 1, -0.5] + rng.normal(size=20)

    def estimate(self, weights, minimum=3, logistic=True):
        if AIPW:
            if logistic:
                patcher = patch.object(
                    model,
                    "propensity_odds_weights",
                    return_value=np.array(weights, dtype=float),
                )
                method = "ps_logistic"
            else:
                patcher = patch.object(model.ULSIFDensityRatio, "fit")
                method = "ulsif"
            with patcher as mocked:
                if not logistic:
                    mocked.return_value.source_weights_ = np.array(weights, dtype=float)
                    mocked.return_value.raw_weight_diagnostics_ = weight_diagnostics(
                        weights
                    )
                    mocked.return_value.ratio_zero_fallback_ = False
                    mocked.return_value._ratio_sigma = 1.0
                    mocked.return_value._ratio_lambda = 0.01
                return model.aipw_estimate(
                    self.x, self.y, self.x, self.y, weight_model=method, min_ess=minimum
                )
        with patch.object(
            model,
            "compute_weights",
            return_value=(np.ones(20), np.array(weights, dtype=float)),
        ):
            return model.ipw_estimate(
                self.x,
                self.y,
                self.x,
                self.y,
                method="propensity" if logistic else "ulsif",
                min_ess=minimum,
            )

    def test_each_weight_rejects_one_positive_and_preserves_measurement(self):
        for logistic in (True, False):
            with (
                self.subTest(logistic=logistic),
                self.assertRaises(InsufficientESSError) as caught,
            ):
                self.estimate([1] + [0] * 19, logistic=logistic)
            self.assertEqual(caught.exception.ess_diagnostics["full"]["ess"], 1)
            self.assertFalse(caught.exception.ess_diagnostics["eligible"])

    def test_threshold_configurable_boundary_and_no_change_when_eligible(self):
        weights = np.r_[np.ones(3), np.zeros(17)]
        after = self.estimate(weights, minimum=3)
        before = self.estimate(weights, minimum=0)
        self.assertEqual(after["estimate"], before["estimate"])
        with self.assertRaises(InsufficientESSError):
            self.estimate(weights, minimum=5)
        for invalid in (-1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                self.estimate(weights, minimum=invalid)

    def test_no_fake_ess_for_zero_density_weights(self):
        if AIPW:
            for minimum in (3, 0):
                density = ULSIFDensityRatio(ratio_cv=False, min_ess=minimum)
                with patch.object(density, "density_ratio", return_value=np.zeros(20)):
                    density.fit(self.x, self.x)
                self.assertEqual(
                    np.count_nonzero(density.source_weights_), 0 if minimum else 20
                )
                self.assertEqual(density.raw_weight_diagnostics_["ess"], 0)
        else:
            for minimum in (3, 0):
                density = model.ULSIF(min_ess=minimum)
                density.basis_ = self.x
                density.coef_ = np.zeros(20)
                density.sigma_ = 1
                self.assertEqual(
                    np.count_nonzero(density.density_ratio(self.x)),
                    0 if minimum else 20,
                )
                self.assertEqual(density.raw_weight_diagnostics_["ess"], 0)

    def test_real_both_weight_algorithms_preserve_eligible_estimates(self):
        np.random.seed(3)
        tx, ty, sx, sy = experiment.generate_data("data_scenario_1", 100, 0.5, 50)
        for method in ("ulsif", "ps_logistic") if AIPW else ("ulsif", "propensity"):
            if AIPW:

                def fn(minimum):
                    return model.aipw_estimate(
                        tx, ty, sx, sy, weight_model=method, min_ess=minimum
                    )
            else:

                def fn(minimum):
                    return model.ipw_estimate(
                        tx, ty, sx, sy, method=method, min_ess=minimum
                    )

            before, after = fn(0), fn(3)
            self.assertEqual(before["estimate"], after["estimate"])
            self.assertGreaterEqual(after["ess"], 3)
            self.assertGreaterEqual(
                after["ess_diagnostics"]["full"]["positive_weights"], 3
            )

    def test_bootstrap_cutoff_and_failed_point(self):
        for failed in (1, 2):
            calls = iter(
                [np.ones(20)]
                + [np.r_[1.0, np.zeros(19)]] * failed
                + [np.ones(20)] * (20 - failed)
            )
            run = run_weighted_bootstrap(
                lambda *a: self.estimate(next(calls)),
                self.x,
                self.y,
                self.x,
                self.y,
                20,
                0,
            )
            self.assertEqual(len(run["boot_errors"]), failed)
            self.assertEqual(len(run["boot_ess_diagnostics"]), 20)
            interval = run["intervals"]["estimate"]
            self.assertEqual(
                interval["error"],
                "" if failed == 1 else "bootstrap_failure_rate_exceeded",
            )
        run = run_weighted_bootstrap(
            lambda *a: self.estimate([1] + [0] * 19),
            self.x,
            self.y,
            self.x,
            self.y,
            2,
            0,
        )
        self.assertEqual(run["point_error_type"], "ESS")
        self.assertEqual(run["intervals"]["estimate"]["error"], "point_estimate_failed")
        row = {**diagnostic_columns(run, "test_"), **variant_columns(run, "test")}
        stats = summarize_trials(pd.DataFrame([row]), "test", 1.0)
        self.assertEqual(stats["n_point_ess_failed"], 1)
        self.assertEqual(stats["n_ci_success"], 0)
        self.assertTrue(np.isnan(stats["coverage"]))

    def test_programming_errors_not_silently_skipped(self):
        def bad(*a):
            raise TypeError("programming error")

        with self.assertRaises(TypeError):
            run_weighted_bootstrap(bad, self.x, self.y, self.x, self.y, 2, 0)

    def test_main_and_nonlinear_keep_real_ess_failures(self):
        config = yaml.safe_load((args.root / "configs/config.yaml").read_text())
        config["hyperparameters"].update(min_ess=100, n_trial=1)
        config["dataset"].update(source_number=20, target_number=50)
        config["scenario"]["data_scenario_id"] = "data_scenario_2"
        if AIPW:
            row = main._run_one_trial(135, config, 2)
            nonlinear = experiment.run_one_trial(
                135,
                "data_scenario_2",
                20,
                0,
                2,
                50,
                dict(weight_model="ulsif", min_ess=100),
            )
            prefix = "aipw"
        else:
            row = main._run_one_trial(
                135, config, "ulsif", 2, main._weight_kwargs(config)
            )
            nonlinear = experiment.run_one_trial(
                135, "data_scenario_2", 20, 0, 2, 50, "ulsif", dict(min_ess=100)
            )
            prefix = "ipw"
        self.assertEqual(row["point_error_type"], "ESS")
        self.assertEqual(row["boot_failed_ESS"], 2)
        self.assertEqual(nonlinear[prefix + "_point_error_type"], "ESS")
        self.assertEqual(nonlinear[prefix + "_boot_failed_ESS"], 2)
        self.assertEqual(
            len(json.loads(nonlinear[prefix + "_boot_ess_diagnostics"])), 2
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            config["output_dir"] = tmp
            Result.save_results([row], config)
            saved = yaml.safe_load(next(Path(tmp).glob("summary*.yaml")).read_text())
            self.assertEqual(saved["statistics"]["n_point_ess_failed"], 1)
            self.assertEqual(saved["feasibility_rule"]["min_ess"], 100)

    def test_resume_protects_rules_and_retains_failed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            meta = dict(
                method="test",
                min_ess=3,
                max_bootstrap_failure_rate=0.05,
                target_n=20,
                bootstraps=2,
                n_trials=1,
            )
            write_run_meta(out, args.root, **meta)
            old = (out / "run_meta.json").read_bytes()
            with self.assertRaises(ValueError):
                write_run_meta(out, args.root, **dict(meta, min_ess=5))
            self.assertEqual((out / "run_meta.json").read_bytes(), old)
            write_run_meta(out, args.root, **dict(meta, n_trials=2))
            run = run_weighted_bootstrap(
                lambda *a: self.estimate([1] + [0] * 19),
                self.x,
                self.y,
                self.x,
                self.y,
                2,
                0,
            )
            row = dict(trial=0, **diagnostic_columns(run, "test_"))
            row.update(variant_columns(run, "test"))
            path = out / "trials_test.csv"
            df, n = load_or_run_trials(path, 1, "test_boot_se", lambda _: row, 1, 0)
            self.assertEqual(n, 1)
            df, n = load_or_run_trials(
                path, 1, "test_boot_se", lambda _: self.fail("rerun"), 1, 0
            )
            self.assertEqual(n, 0)
            resumed_stats = summarize_trials(df, "test", 1.0)
            self.assertEqual(resumed_stats["n_point_ess_failed"], 1)
            self.assertEqual(resumed_stats["n_ci_success"], 0)
            self.assertTrue(pd.isna(df.test_estimate.iloc[0]))
            self.assertEqual(df.test_point_error_type.iloc[0], "ESS")
            self.assertEqual(len(list((out / "checkpoints").rglob("trial*.json"))), 1)

    def test_cli_config_override_and_mixing_guard(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            out = Path(tmp) / "run"
            cfg = Path(tmp) / "config.yaml"
            cfg.write_text(
                yaml.safe_dump(
                    dict(
                        hyperparameters=dict(min_ess=5, max_bootstrap_failure_rate=0.2)
                    )
                )
            )
            argv = [
                "experiment",
                "--config",
                str(cfg),
                "--min-ess",
                "100",
                "--trials",
                "1",
                "--bootstraps",
                "2",
                "--source-ns",
                "20",
                "--target-n",
                "50",
                "--scenarios",
                "data_scenario_2",
                "--gammas",
                "0",
                "--n-jobs",
                "1",
                "--out-dir",
                str(out),
            ]
            with patch.object(sys, "argv", argv):
                experiment.main()
            meta = json.loads((out / "run_meta.json").read_text())
            self.assertEqual(meta["min_ess"], 100)
            self.assertEqual(meta["max_bootstrap_failure_rate"], 0.2)
            with patch.object(sys, "argv", argv):
                experiment.main()
            argv[argv.index("100")] = "101"
            before = (out / "run_meta.json").read_bytes()
            with patch.object(sys, "argv", argv), self.assertRaises(ValueError):
                experiment.main()
            self.assertEqual((out / "run_meta.json").read_bytes(), before)


unittest.main(argv=["checks"], verbosity=2)
