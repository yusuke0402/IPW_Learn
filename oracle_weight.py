"""Raw Gaussian density ratios for reference estimators; x excludes the intercept."""

from collections.abc import Mapping

import numpy as np


def _gaussian_logpdf(x, mean, cov):
    try:
        x = np.asarray(x, dtype=float)
        mean = np.asarray(mean, dtype=float)
        cov = np.asarray(cov, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Oracle inputs must be numeric arrays.") from exc
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("Oracle covariates must be a two-dimensional array without the intercept.")
    dimension = x.shape[1]
    if mean.shape != (dimension,) or cov.shape != (dimension, dimension):
        raise ValueError("Oracle mean and covariance dimensions must match the covariates.")
    if not (np.isfinite(x).all() and np.isfinite(mean).all() and np.isfinite(cov).all()):
        raise ValueError("Oracle covariates, means, and covariances must be finite.")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Oracle covariance matrices must be symmetric.")
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Oracle covariance matrices must be positive definite.") from exc
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            standardized = np.linalg.solve(chol, (x - mean).T)
            maha = np.sum(standardized ** 2, axis=0)
            logdet = 2.0 * np.sum(np.log(np.diag(chol)))
            result = -0.5 * (maha + logdet + dimension * np.log(2.0 * np.pi))
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError("Oracle Gaussian log densities are not representable as finite floats.") from exc
    if not np.isfinite(result).all():
        raise ValueError("Oracle Gaussian log densities must be finite.")
    return result


def _require_domains(means, covariances):
    if (not isinstance(means, Mapping) or not isinstance(covariances, Mapping)
            or any(domain not in means or domain not in covariances for domain in ("target", "source"))):
        raise ValueError("Oracle means and covariances need target and source entries.")


def oracle_density_ratio(x, means, covariances):
    """Return raw p_target(x) / p_source(x); underflow is zero and overflow fails."""
    _require_domains(means, covariances)
    log_target = _gaussian_logpdf(x, means["target"], covariances["target"])
    log_source = _gaussian_logpdf(x, means["source"], covariances["source"])
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            weights = np.exp(log_target - log_source)
    except FloatingPointError as exc:
        raise ValueError("Oracle density ratios are not representable as finite floats.") from exc
    if not np.isfinite(weights).all():
        raise ValueError("Oracle density ratios must be finite.")
    return weights


def gaussian_oracle_metadata(means, covariances):
    """Validate and serialize the true distributions used by an oracle run."""
    _require_domains(means, covariances)
    try:
        mean_arrays = {domain: np.asarray(means[domain], dtype=float) for domain in ("target", "source")}
    except (TypeError, ValueError) as exc:
        raise ValueError("Oracle means must be numeric vectors.") from exc
    if any(mean.ndim != 1 or len(mean) < 1 for mean in mean_arrays.values()):
        raise ValueError("Oracle means must be nonempty vectors.")
    dimension = len(mean_arrays["target"])
    for domain in ("target", "source"):
        _gaussian_logpdf(np.empty((0, dimension)), mean_arrays[domain], covariances[domain])
    return dict(distribution="multivariate_normal", ratio="p_target/p_source", intercept_in_density=False,
                means={domain: mean_arrays[domain].tolist() for domain in ("target", "source")},
                covariances={domain: np.asarray(covariances[domain], dtype=float).tolist()
                             for domain in ("target", "source")})
