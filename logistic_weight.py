"""Common logistic membership model for IWL, IPW and self-normalized AIPW.

Inputs exclude the design intercept. C selection uses only group labels and X;
it never uses outcomes, ESS or simulation truth. Kept identical in all repos.
"""

import warnings
import inspect

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

DEFAULT_C_GRID = tuple(np.logspace(-4, 4, 17))
DEFAULT_CV_FOLDS = 5
DEFAULT_CV_SEED = 2024
DEFAULT_TOL = 1e-8
DEFAULT_SOLVER = "newton-cholesky"


def fit_membership_logistic(target_x, source_x, *, cv=True, C=1.0,
                            penalty="l2", C_grid=None, cv_folds=DEFAULT_CV_FOLDS,
                            cv_seed=DEFAULT_CV_SEED, max_iter=1000):
    tx, sx = np.asarray(target_x, dtype=float), np.asarray(source_x, dtype=float)
    if tx.ndim != 2 or sx.ndim != 2 or tx.shape[1] != sx.shape[1]:
        raise ValueError("Logistic inputs must be matrices with matching features.")
    if min(len(tx), len(sx)) < 1 or not np.isfinite(tx).all() or not np.isfinite(sx).all():
        raise ValueError("Logistic inputs must be nonempty and finite.")
    if penalty not in ("l2", "none"):
        raise ValueError("Logistic penalty must be 'l2' or 'none'.")
    X = np.vstack([tx, sx])
    labels = np.concatenate([np.ones(len(tx)), np.zeros(len(sx))])
    if cv and penalty == "l2":
        if not isinstance(cv_folds, (int, np.integer)) or cv_folds < 2:
            raise ValueError("Logistic CV requires at least two folds.")
        folds = min(cv_folds, len(tx), len(sx))
        if folds < 2:
            raise ValueError("Logistic CV needs at least two samples in each group.")
        grid = np.asarray(DEFAULT_C_GRID if C_grid is None else C_grid, dtype=float)
        if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or (grid <= 0).any():
            raise ValueError("C grid must contain finite positive numbers.")
        grid = np.unique(grid)
        split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cv_seed)
        # sklearn 1.8+ exposes a transition switch for fitted attributes.
        # Keep scores in the same shape while allowing IPW's sklearn >=1.6.
        compatibility = ({"use_legacy_attributes": True, "l1_ratios": (0.0,)}
                         if "use_legacy_attributes" in inspect.signature(LogisticRegressionCV).parameters else {})
        model = LogisticRegressionCV(
            Cs=grid, cv=split, scoring="neg_log_loss", solver=DEFAULT_SOLVER,
            tol=DEFAULT_TOL, max_iter=max_iter, n_jobs=1, refit=True,
            **compatibility,
        )
    else:
        if penalty == "l2" and (not np.isfinite(C) or C <= 0):
            raise ValueError("Fixed C must be finite and positive.")
        model = LogisticRegression(C=np.inf if penalty == "none" else C,
                                   solver=DEFAULT_SOLVER, tol=DEFAULT_TOL, max_iter=max_iter)
        folds = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(X, labels)
        except ConvergenceWarning as exc:
            raise RuntimeError(f"Logistic did not converge: {exc}") from exc
    selected = float(model.C_[0]) if folds else float(model.C)
    model.weight_selection_ = dict(
        cv=bool(folds), C=selected, cv_folds=folds, cv_seed=cv_seed,
        scoring="neg_log_loss" if folds else None,
        C_grid=model.Cs_.tolist() if folds else [],
        cv_log_loss=(-model.scores_[1.0].reshape(folds, -1).mean(axis=0)).tolist() if folds else [],
        solver=DEFAULT_SOLVER, tol=DEFAULT_TOL, max_iter=max_iter,
        probability_clip=1e-12,
    )
    return model


def source_odds(model, source_x, clip=1e-12):
    if clip is None:
        clip = 1e-12
    if not np.isfinite(clip) or not 0 < clip < .5:
        raise ValueError("Propensity clip must be between zero and 0.5.")
    e = np.clip(model.predict_proba(source_x)[:, 1], clip, 1 - clip)
    return e / (1 - e)
