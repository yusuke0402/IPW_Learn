"""Shared handling of numerical fit failures and bootstrap intervals."""
import numpy as np

NUMERICAL_ERRORS = (ValueError, RuntimeError, np.linalg.LinAlgError)
DEFAULT_MIN_ESS = 3.0
DEFAULT_MAX_BOOTSTRAP_FAILURE_RATE = 0.05


class InsufficientESSError(ValueError):
    """The historical sample lacks the prespecified effective sample size."""


def validate_min_ess(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("min_ess must be finite and nonnegative.")
    return value


def assess_ess(weights, min_ess=DEFAULT_MIN_ESS, **metadata):
    """Screen historical weights and retain their measured diagnostics on failure."""
    minimum = validate_min_ess(min_ess)
    full = weight_diagnostics(weights)
    diagnostics = dict(min_ess=minimum, full=full, cv_folds=[],
                       cv_min_train_ess=None, eligible=full["ess"] >= minimum,
                       **metadata)
    if not diagnostics["eligible"] or full["positive_weights"] == 0:
        diagnostics.update(eligible=False, failure_scope="full_sample")
        exc = InsufficientESSError(
            f"Historical ESS={full['ess']:.6g} < minimum={minimum:g}, "
            f"or no positive weight ({full['positive_weights']}/{full['n']}).")
        exc.ess_diagnostics = diagnostics
        raise exc
    return diagnostics


def weight_diagnostics(weights):
    """Kish ESS of nonnegative weights, invariant to their common scale.

    Scaling before summing/squaring avoids overflow and underflow. All-zero
    weights have ESS=0 by convention and are never replaced here.
    """
    w = np.asarray(weights, dtype=float).ravel()
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Regression weights must be finite and nonnegative.")
    positive = int(np.count_nonzero(w > 0))
    if positive:
        scaled = w / w.max()
        total = float(scaled.sum())
        ess = min(float(total**2 / (scaled @ scaled)), float(positive))
        maximum = 1.0 / total
    else:
        ess, maximum = 0.0, 0.0
    return dict(n=int(w.size), positive_weights=positive, ess=ess,
                max_normalized_weight=maximum)


def compact_ess_diagnostics(diagnostics):
    """Small per-replicate record, including folds even when fitting fails."""
    full = diagnostics.get("full") or {}
    folds = diagnostics.get("cv_folds", [])
    return dict(ess=full.get("ess"), positive_weights=full.get("positive_weights"),
                max_normalized_weight=full.get("max_normalized_weight"),
                min_ess=diagnostics.get("min_ess"), eligible=diagnostics.get("eligible"),
                cv_split=diagnostics.get("cv_split"),
                cv_requested_folds=diagnostics.get("cv_requested_folds"),
                cv_actual_folds=diagnostics.get("cv_actual_folds"),
                cv_min_train_ess=diagnostics.get("cv_min_train_ess"),
                cv_train_ess=[fold["train"]["ess"] for fold in folds],
                cv_train_positive_weights=[fold["train"]["positive_weights"] for fold in folds],
                cv_validation_positive_weights=[fold["validation"]["positive_weights"] for fold in folds])


def error_category(exc):
    if isinstance(exc, InsufficientESSError):
        return "ESS"
    message = str(exc)
    if "did not converge" in message:
        return "nonconvergence"
    if "KKT" in message:
        return "KKT"
    if "weights" in message:
        return "weights"
    return "numerical"


def error_message(exc):
    return f"{type(exc).__name__}: {exc}"


def bootstrap_interval(point, estimates, *, requested=None, max_failure_rate=1.0):
    """Reject CIs for failed points, too few successes, or excess failures.

    Failed replicates must be recorded separately by the caller. A bootstrap SE
    can still be computed when the original point estimate failed. Entry points
    pass the prespecified failure limit; the default retains legacy behavior.
    """
    values = np.asarray(estimates, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap estimates must be finite.")
    if not np.isfinite(max_failure_rate) or not 0 <= max_failure_rate <= 1:
        raise ValueError("max_bootstrap_failure_rate must be between 0 and 1.")
    requested = len(values) if requested is None else requested
    if not isinstance(requested, (int, np.integer)) or requested < len(values):
        raise ValueError("Requested bootstrap count must be an integer >= successes.")
    failure_rate = (requested - len(values)) / requested if requested else 0.0
    variance = float(values.var(ddof=1)) if len(values) >= 2 else float("nan")
    result = dict(variance=variance, se=float(np.sqrt(variance)),
                  lower=float("nan"), upper=float("nan"), error="", failure_rate=failure_rate)
    if not np.isfinite(point):
        result["error"] = "point_estimate_failed"
    elif len(values) < 2:
        result["error"] = "insufficient_bootstrap_success"
    elif failure_rate > max_failure_rate:
        result["error"] = "bootstrap_failure_rate_exceeded"
    else:
        result["lower"], result["upper"] = map(float, np.percentile(values, [2.5, 97.5]))
    return result
