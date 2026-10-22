"""ESS-screened full-pipeline bootstrap and explicit failure reporting.

Estimators may opt in to receiving original observation IDs in every fit.
"""

import json

import numpy as np

from experiment_diagnostics import (
    DEFAULT_MAX_BOOTSTRAP_FAILURE_RATE,
    DEFAULT_MIN_ESS,
    validate_min_ess,
    NUMERICAL_ERRORS,
    bootstrap_interval,
    compact_ess_diagnostics,
    error_category,
    error_message,
)


def run_weighted_bootstrap(
    estimator,
    tx,
    ty,
    sx,
    sy,
    requested,
    seed,
    estimate_keys=("estimate",),
    max_failure_rate=DEFAULT_MAX_BOOTSTRAP_FAILURE_RATE,
    min_ess=DEFAULT_MIN_ESS,
    bootstrap_estimator=None,
    pass_observation_ids=False,
):
    """Resample rows and optionally forward their original IDs for grouped model CV."""
    min_ess = validate_min_ess(min_ess)
    if not isinstance(requested, (int, np.integer)) or requested < 0:
        raise ValueError("Bootstrap count must be a nonnegative integer.")
    # Validate the rule before running any fit.
    bootstrap_interval(0, [], requested=requested, max_failure_rate=max_failure_rate)
    errors, diagnostics, values = [], [], {key: [] for key in estimate_keys}

    def fit(a, b, c, d, *, bootstrap=False, target_ids=None, source_ids=None):
        callback = bootstrap_estimator if bootstrap and bootstrap_estimator is not None else estimator
        ids = dict(target_ids=target_ids, source_ids=source_ids) if pass_observation_ids else {}
        result = callback(a, b, c, d, **ids)
        if not all(np.isfinite(result[key]) for key in estimate_keys):
            exc = ValueError("Point estimate must be finite.")
            exc.ess_diagnostics = result.get("ess_diagnostics", {})
            raise exc
        return result

    point, point_error, point_type, point_diagnostics = {}, "", "", {}
    try:
        point = fit(tx, ty, sx, sy, target_ids=np.arange(len(tx)), source_ids=np.arange(len(sx)))
        point_diagnostics = point.get("ess_diagnostics", {})
    except NUMERICAL_ERRORS as exc:
        point_error, point_type = error_message(exc), error_category(exc)
        point_diagnostics = getattr(exc, "ess_diagnostics", {})
    point_diagnostics.setdefault("min_ess", min_ess)
    rng = np.random.default_rng(seed)
    for index in range(requested):
        ti = rng.choice(len(tx), size=len(tx), replace=True)
        si = rng.choice(len(sx), size=len(sx), replace=True)
        try:
            result = fit(tx[ti], ty[ti], sx[si], sy[si], bootstrap=True,
                         target_ids=ti, source_ids=si)
            diag = result.get("ess_diagnostics", {})
            for key in estimate_keys:
                values[key].append(result[key])
        except NUMERICAL_ERRORS as exc:
            diag = getattr(exc, "ess_diagnostics", {})
            errors.append(
                dict(
                    iteration=index,
                    error=error_message(exc),
                    type=error_category(exc),
                    ess_diagnostics=diag,
                )
            )
        diag.setdefault("min_ess", min_ess)
        diagnostics.append(dict(iteration=index, **compact_ess_diagnostics(diag)))
    intervals = {
        key: bootstrap_interval(
            point.get(key, np.nan),
            values[key],
            requested=requested,
            max_failure_rate=max_failure_rate,
        )
        for key in estimate_keys
    }
    return dict(
        point=point,
        point_error=point_error,
        point_error_type=point_type,
        ess_diagnostics=point_diagnostics,
        intervals=intervals,
        boot_n=len(values[estimate_keys[0]]),
        boot_requested=requested,
        boot_errors=errors,
        boot_ess_diagnostics=diagnostics,
        max_bootstrap_failure_rate=max_failure_rate,
    )


def diagnostic_columns(run, prefix=""):
    d = run["ess_diagnostics"]
    full = d.get("full", {})
    errors = run["boot_errors"]
    first_interval = next(iter(run["intervals"].values()))
    return {
        prefix + key: value
        for key, value in dict(
            ess=full.get("ess", np.nan),
            positive_weights=full.get("positive_weights", np.nan),
            max_normalized_weight=full.get("max_normalized_weight", np.nan),
            min_ess=d.get("min_ess", np.nan),
            ess_eligible=d.get("eligible"),
            point_error=run["point_error"],
            point_error_type=run["point_error_type"],
            ess_diagnostics=json.dumps(d, ensure_ascii=False),
            boot_requested=run["boot_requested"],
            boot_n=run["boot_n"],
            boot_failed=len(errors),
            boot_failed_ESS=sum(e["type"] == "ESS" for e in errors),
            boot_failed_positive_weights=sum(e["type"] == "positive_weights" for e in errors),
            boot_failure_rate=first_interval["failure_rate"],
            max_bootstrap_failure_rate=run["max_bootstrap_failure_rate"],
            boot_errors=json.dumps(errors, ensure_ascii=False),
            boot_ess_diagnostics=json.dumps(
                run["boot_ess_diagnostics"], ensure_ascii=False
            ),
        ).items()
    }


def variant_columns(run, prefix, key="estimate"):
    interval = run["intervals"][key]
    return {
        prefix + suffix: value
        for suffix, value in dict(
            _estimate=run["point"].get(key, np.nan),
            _boot_se=interval["se"],
            _ci_lower=interval["lower"],
            _ci_upper=interval["upper"],
            _ci_error=interval["error"],
            _boot_n=run["boot_n"],
        ).items()
    }


def main_columns(run):
    interval = run["intervals"]["estimate"]
    return dict(
        estimate_value=run["point"].get("estimate", np.nan),
        bootstrap_variance=interval["variance"],
        bootstrap_se=interval["se"],
        bootstrap_n=run["boot_n"],
        estimate_95ci_lower=interval["lower"],
        estimate_95ci_upper=interval["upper"],
        ci_error=interval["error"],
        **diagnostic_columns(run),
    )


def summarize_trials(df, prefix, true_value, diagnostics_prefix=None):
    """Performance conditional on usable estimates; always show denominators."""
    import pandas as pd

    dp = diagnostics_prefix or prefix
    est = pd.to_numeric(df[f"{prefix}_estimate"], errors="raise").astype(float)
    est = est.where(np.isfinite(est))
    lo = pd.to_numeric(df[f"{prefix}_ci_lower"], errors="raise").astype(float)
    hi = pd.to_numeric(df[f"{prefix}_ci_upper"], errors="raise").astype(float)
    valid = np.isfinite(est) & np.isfinite(lo) & np.isfinite(hi)
    se = pd.to_numeric(df[f"{prefix}_boot_se"], errors="raise").astype(float)
    normal = valid & np.isfinite(se)
    n_success = int(est.notna().sum())
    point_types = df.get(
        f"{dp}_point_error_type", pd.Series("", index=df.index)
    ).fillna("")
    errors = df.get(f"{prefix}_ci_error", pd.Series("", index=df.index)).fillna("")
    boot_requested = df[f"{dp}_boot_requested"].sum()
    boot_failed = int(df[f"{dp}_boot_failed"].sum())
    boot_ess = int(df[f"{dp}_boot_failed_ESS"].sum())
    stats = dict(
        mean=float(est.mean()),
        bias=float(est.mean() - true_value),
        variance=float(est.var(ddof=1)),
        sd=float(est.std(ddof=1)),
        rmse=float(np.sqrt(((est - true_value) ** 2).mean())),
        coverage=float(((lo[valid] <= true_value) & (true_value <= hi[valid])).mean()),
        ci_width=float((hi[valid] - lo[valid]).mean()),
        coverage_normal=float(
            (
                (est[normal] - 1.96 * se[normal] <= true_value)
                & (true_value <= est[normal] + 1.96 * se[normal])
            ).mean()
        ),
        ci_width_normal=float((3.92 * se[normal]).mean()),
        n_point_success=n_success,
        n_point_failed=len(df) - n_success,
        point_failure_rate=(len(df) - n_success) / len(df) if len(df) else np.nan,
        n_point_ess_failed=int((point_types == "ESS").sum()),
        point_ess_failure_rate=float((point_types == "ESS").mean()),
        n_ci_success=int(valid.sum()),
        n_ci_normal_success=int(normal.sum()),
        n_ci_failed=int((~valid).sum()),
        n_ci_bootstrap_failure_rate_exceeded=int(
            (errors == "bootstrap_failure_rate_exceeded").sum()
        ),
        boot_n_mean=float(df[f"{prefix}_boot_n"].mean()),
        boot_n_min=int(df[f"{prefix}_boot_n"].min()),
        bootstrap_requested=int(boot_requested),
        bootstrap_failed=boot_failed,
        bootstrap_failed_ESS=boot_ess,
        bootstrap_failed_positive_weights=int(df.get(
            f"{dp}_boot_failed_positive_weights", pd.Series(0, index=df.index)).sum()),
        bootstrap_failure_rate=float(boot_failed / boot_requested)
        if boot_requested
        else 0.0,
        bootstrap_ess_failure_rate=float(boot_ess / boot_requested)
        if boot_requested
        else 0.0,
        ess_median=float(df[f"{dp}_ess"].median()),
        min_ess=float(df[f"{dp}_min_ess"].iloc[0]),
        max_bootstrap_failure_rate=float(
            df[f"{dp}_max_bootstrap_failure_rate"].iloc[0]
        ),
    )
    return stats
