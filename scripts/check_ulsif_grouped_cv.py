"""Check observation-group CV, training-only bases, and IPW bootstrap integration.

uv run scripts/check_ulsif_grouped_cv.py
uv run scripts/check_ulsif_grouped_cv.py --aipw-root ../AIPW-Learn
"""

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main
from experiment_execution import run_weighted_bootstrap
from experiment_io import write_run_meta
from scripts import nonlinear_gamma_ipw as experiment
from weight import ULSIF, compute_weights

AIPW = None


class GroupedCVChecks(unittest.TestCase):
    def setUp(self):
        # Match the experiment entry points' filter for Accelerate matmul warnings.
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        rng = np.random.default_rng(41)
        self.tx = np.column_stack([np.ones(24), rng.normal(size=(24, 2))])
        self.sx = np.column_stack([np.ones(35), rng.normal(size=(35, 2))])

    def assert_no_group_leak(self, folds, target_ids, source_ids):
        for tr_t, va_t, tr_s, va_s in folds:
            for train, valid, ids in ((tr_t, va_t, target_ids), (tr_s, va_s, source_ids)):
                self.assertTrue(set(ids[train]).isdisjoint(ids[valid]))
                self.assertGreater(len(train), 0)
                self.assertGreater(len(valid), 0)
                np.testing.assert_array_equal(np.sort(np.r_[train, valid]), np.arange(len(ids)))
        for index, ids in ((1, target_ids), (3, source_ids)):
            np.testing.assert_array_equal(
                np.sort(np.concatenate([fold[index] for fold in folds])), np.arange(len(ids)))

    def test_copies_stay_in_one_fold_and_natural_ties_have_distinct_ids(self):
        ti = np.repeat([101, 203, 307, 409, 503, 607], [4, 1, 2, 3, 1, 2])
        si = np.repeat([11, 17, 23, 31, 41, 47, 59], [1, 3, 2, 1, 4, 1, 2])
        tx = np.column_stack([np.ones(len(ti)), ti, ti ** 2])
        sx = np.column_stack([np.ones(len(si)), si, si ** 2])
        model = ULSIF()
        folds = model._make_cv_folds(tx, sx, target_ids=ti, source_ids=si)
        self.assertEqual(len(folds), 5)
        self.assert_no_group_leak(folds, ti, si)
        ti, si = np.arange(10), np.arange(12)
        folds = model._make_cv_folds(np.ones((10, 3)), np.ones((12, 3)),
                                     target_ids=ti, source_ids=si)
        self.assertEqual(len(folds), 5)
        self.assert_no_group_leak(folds, ti, si)

    def test_missing_ids_group_identical_x_and_reduce_folds(self):
        tx = np.repeat(np.arange(3)[:, None], [4, 2, 3], axis=0)
        sx = np.repeat(np.arange(4)[:, None], [2, 1, 3, 2], axis=0)
        model = ULSIF()
        folds = model._make_cv_folds(tx, sx)
        self.assertEqual(len(folds), 3)
        self.assert_no_group_leak(folds, tx.ravel(), sx.ravel())
        for target, source in ((tx[:1], sx), (tx, sx[:1])):
            with self.assertRaisesRegex(ValueError, "two observation groups"):
                model._make_cv_folds(target, source)
        for name, x in (("target_ids", tx), ("source_ids", sx)):
            for ids in (np.arange(len(x) - 1), np.zeros((len(x), 1)),
                        np.r_[np.nan, np.arange(len(x) - 1)],
                        np.r_[np.inf, np.arange(len(x) - 1)]):
                with self.subTest(name=name, ids=ids), self.assertRaises(ValueError):
                    model._make_cv_folds(tx, sx, **{name: ids})

    def test_cv_bases_use_training_target_once_and_preserve_global_rng(self):
        model = ULSIF()
        model.MAX_BASIS = 8
        folds = model._make_cv_folds(self.tx, self.sx)
        selected = []
        original = model._select_basis

        def capture(target, *args, **kwargs):
            basis = original(target, *args, **kwargs)
            selected.append((target.copy(), basis.copy()))
            return basis

        np.random.seed(19)
        expected = np.random.random(5)
        np.random.seed(19)
        with patch.object(model, "_select_basis", side_effect=capture):
            model._select_hyperparams(self.tx, self.sx, self.tx,
                model._median_pairwise_distance(np.vstack([self.tx, self.sx])))
        np.testing.assert_array_equal(np.random.random(5), expected)
        self.assertEqual(len(selected), len(folds))
        for (train, basis), (tr_t, va_t, _, _) in zip(selected, folds):
            np.testing.assert_array_equal(train, self.tx[tr_t])
            self.assertEqual(len(basis), 8)
            self.assertTrue(set(map(tuple, basis)).issubset(set(map(tuple, self.tx[tr_t]))))
            self.assertTrue(set(map(tuple, basis)).isdisjoint(set(map(tuple, self.tx[va_t]))))

    def test_selection_matches_independent_training_only_risk(self):
        tx = np.array([[0., 0.], [0., 0.], [1., 2.], [3., 1.], [3., 1.], [4., 4.]])
        sx = np.array([[.5, -.5], [.5, -.5], [1., 1.], [1., 1.],
                       [2., 3.], [4., 2.], [4., 2.], [5., 4.]])
        model = ULSIF()
        model.SIGMA_FACTORS = (.5, 1.5)
        model.LAMBDA_GRID = (.001, .1)
        all_x = np.vstack([tx, sx])
        distances = np.linalg.norm(all_x[:, None] - all_x[None, :], axis=2)
        anchor = np.median(distances[np.triu_indices(len(all_x), k=1)])
        folds = model._make_cv_folds(tx, sx)
        risks = []
        for sigma in np.asarray(model.SIGMA_FACTORS) * anchor:
            for lam in model.LAMBDA_GRID:
                scores = []
                for tr_t, va_t, tr_s, va_s in folds:
                    basis = tx[tr_t]

                    def kernel(x):
                        return np.exp(-np.sum((x[:, None] - basis[None, :]) ** 2, axis=2)
                                      / (2 * sigma ** 2))

                    ks, kt = kernel(sx[tr_s]), kernel(tx[tr_t])
                    coef = np.linalg.solve(ks.T @ ks / len(tr_s) + lam * np.eye(len(basis)),
                                           kt.mean(axis=0))
                    ws = np.maximum(kernel(sx[va_s]) @ coef, 0)
                    wt = np.maximum(kernel(tx[va_t]) @ coef, 0)
                    scores.append(.5 * np.mean(ws ** 2) - np.mean(wt))
                risks.append((np.mean(scores), sigma, lam))
        _, sigma, lam = min(risks, key=lambda item: item[0])
        actual = model._select_hyperparams(tx, sx, tx, anchor)
        np.testing.assert_allclose(actual, (sigma, lam), rtol=1e-12)

    def test_original_and_bootstrap_weights_match_aipw(self):
        if AIPW is None:
            self.skipTest("Pass --aipw-root for an independent implementation comparison.")
        rng = np.random.default_rng(6)
        for bootstrap, cap in ((False, 100), (True, 100), (True, 8)):
            ti = rng.choice(len(self.tx), len(self.tx)) if bootstrap else np.arange(len(self.tx))
            si = rng.choice(len(self.sx), len(self.sx)) if bootstrap else np.arange(len(self.sx))
            tx, sx = self.tx[ti], self.sx[si]
            for with_ids in (False, True):
                with self.subTest(bootstrap=bootstrap, basis_cap=cap, with_ids=with_ids):
                    ids = dict(target_ids=ti, source_ids=si) if with_ids else {}
                    ipw, aipw = ULSIF(), AIPW()
                    # Compare CV with identical grids, preserving this repository's candidate settings.
                    aipw.RATIO_SIGMA_FACTORS = ipw.SIGMA_FACTORS
                    aipw.RATIO_LAMBDA_GRID = ipw.LAMBDA_GRID
                    ipw.MAX_BASIS = aipw.RATIO_MAX_BASIS = cap
                    np.random.seed(79)
                    ipw.fit(tx, sx, bootstrap=bootstrap, **ids)
                    np.random.seed(79)
                    aipw.fit(tx, sx, bootstrap=bootstrap, **ids)
                    self.assertEqual(ipw.sigma_, aipw._ratio_sigma)
                    self.assertEqual(ipw.lambda_, aipw._ratio_lambda)
                    np.testing.assert_array_equal(ipw.basis_, aipw._ratio_basis)
                    np.testing.assert_allclose(ipw.coef_, aipw._ratio_coef.ravel(), rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(ipw.density_ratio(sx), aipw.source_weights_, rtol=1e-12, atol=1e-12)

    def test_main_and_nonlinear_forward_original_ids_and_reselect_every_fit(self):
        config = yaml.safe_load((ROOT / "configs/config.yaml").read_text())
        config["hyperparameters"].update(n_features=2, min_ess=3)
        config["dataset"].update(target_number=24, source_number=35)
        original_fit = ULSIF.fit
        original_select = ULSIF._select_hyperparams
        for entry in ("main", "nonlinear"):
            records, selections = [], []
            originals = {}

            def capture_fit(model, tx, sx, **kwargs):
                ti, si = kwargs["target_ids"], kwargs["source_ids"]
                bootstrap = kwargs["bootstrap"]
                if not bootstrap:
                    originals.update(target=tx.copy(), source=sx.copy())
                    np.testing.assert_array_equal(ti, np.arange(len(tx)))
                    np.testing.assert_array_equal(si, np.arange(len(sx)))
                np.testing.assert_array_equal(tx, originals["target"][ti])
                np.testing.assert_array_equal(sx, originals["source"][si])
                folds = model._make_cv_folds(tx, sx, target_ids=ti, source_ids=si)
                self.assert_no_group_leak(folds, ti, si)
                records.append((bootstrap, len(np.unique(ti)), len(ti)))
                return original_fit(model, tx, sx, **kwargs)

            def capture_selection(model, *args, **kwargs):
                selections.append(kwargs)
                return original_select(model, *args, **kwargs)

            with patch.object(ULSIF, "fit", new=capture_fit), \
                 patch.object(ULSIF, "_select_hyperparams", new=capture_selection):
                if entry == "main":
                    main._run_one_trial(2, config, "ulsif", 2, main._weight_kwargs(config))
                else:
                    experiment.run_one_trial(2, "data_scenario_1", 35, .5, 2, 24,
                                              "ulsif", dict(min_ess=3))
            self.assertEqual([r[0] for r in records], [False, True, True])
            self.assertTrue(all(unique < size for _, unique, size in records[1:]))
            self.assertEqual(len(selections), 3)

    def test_bootstrap_helper_keeps_legacy_callbacks_and_optional_ids(self):
        tx = np.column_stack([np.ones(6), np.arange(6)])
        sx = np.column_stack([np.ones(7), np.arange(7)])
        run = run_weighted_bootstrap(lambda a, b, c, d: {"estimate": 0.},
                                     tx, tx[:, 1], sx, sx[:, 1], 2, 13)
        self.assertEqual(run["boot_n"], 2)
        seen = []

        def with_ids(a, b, c, d, **ids):
            seen.append(ids)
            np.testing.assert_array_equal(a[:, 1], ids["target_ids"])
            np.testing.assert_array_equal(c[:, 1], ids["source_ids"])
            return {"estimate": 0.}

        run_weighted_bootstrap(with_ids, tx, tx[:, 1], sx, sx[:, 1], 2, 13,
                               pass_observation_ids=True)
        self.assertEqual(len(seen), 3)

    def test_uniform_propensity_and_oracle_ignore_ulsif_ids(self):
        means = dict(target=np.zeros(2), source=np.ones(2))
        covs = dict(target=np.eye(2), source=np.eye(2))
        for method in ("uniform", "propensity", "oracle"):
            with self.subTest(method=method), patch("weight.propensityscore", return_value=(
                    np.full(len(self.tx), .4), np.full(len(self.sx), .3))):
                before = compute_weights(method, self.tx, self.sx, means=means, covariances=covs)
                after = compute_weights(method, self.tx, self.sx, means=means, covariances=covs,
                                        target_ids=np.arange(len(self.tx)), source_ids=np.arange(len(self.sx)))
            for a, b in zip(before, after):
                np.testing.assert_array_equal(a, b)


    def test_metadata_refuses_previous_cv_policy_without_modifying_saved_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            meta = dict(method="IPW", weight="ulsif", ratio_cv_policy=ULSIF.CV_POLICY,
                        ratio_sigma_anchor=ULSIF.SIGMA_ANCHOR, n_trials=1, target_n=24,
                        bootstraps=2, min_ess=3, max_bootstrap_failure_rate=.05)
            write_run_meta(out, ROOT, **meta)
            path = out / "run_meta.json"
            saved = json.loads(path.read_text())
            self.assertEqual(saved["resume_contract"]["ratio_cv_policy"], ULSIF.CV_POLICY)
            saved["resume_contract"].pop("ratio_cv_policy")
            path.write_text(json.dumps(saved))
            previous = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "new output directory"):
                write_run_meta(out, ROOT, **meta)
            self.assertEqual(path.read_bytes(), previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aipw-root", type=Path)
    args, rest = parser.parse_known_args()
    if args.aipw_root:
        spec = importlib.util.spec_from_file_location(
            "aipw_density_ratio_reference", args.aipw_root.resolve() / "density_ratio.py")
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        AIPW = reference.ULSIFDensityRatio
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)
