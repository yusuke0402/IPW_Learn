"""Thin, process-isolated launch adapters for the ten frequentist methods.

All estimation, density-ratio fitting, CV, bootstrap resampling and eligibility
rules are delegated to the pinned repository's existing ``run_one_trial``.
IWL uses all target observations, alpha=1, beta=0 and fixed lambda1=0 (WLS).
AIPW and IPW use their self-normalized estimates. The true treatment effect is 1.
The sole missing entry point is IWL oracle: an existing OracleWeightedLearn
factory temporarily replaces the IWL module's learner reference.

Run each method in a separate worker process. This is important because these
repositories use the same unqualified Python module names for different files.
This adapter never changes repository source files or writes study outputs.
"""

from contextlib import contextmanager
import functools
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


METHODS = {
    "iwl_ulsif_l1none": ("iwl", "ulsif"),
    "iwl_oracle_l1none": ("iwl", "oracle"),
    "iwl_logistic_l1none": ("iwl", "ps_logistic"),
    "aipw_sn_ulsif": ("aipw", "ulsif"),
    "aipw_sn_oracle": ("aipw", "oracle"),
    "aipw_sn_logistic": ("aipw", "ps_logistic"),
    "ipw_sn_ulsif": ("ipw", "ulsif"),
    "ipw_sn_oracle": ("ipw", "oracle"),
    "ipw_sn_logistic": ("ipw", "propensity"),
    "g_computation": ("gcomp", "none"),
}
ENTRY_POINTS = {
    "iwl": "scripts/nonlinear_gamma_experiment.py",
    "aipw": "scripts/nonlinear_gamma_aipw.py",
    "ipw": "scripts/nonlinear_gamma_ipw.py",
    "gcomp": "scripts/nonlinear_gamma_gcomp.py",
}
_BOUND = None
_MODULE = None


def _load_source(method, source):
    global _BOUND, _MODULE
    source = Path(source).resolve(strict=True)
    binding = (method, str(source))
    if _BOUND is not None:
        if binding != _BOUND:
            raise RuntimeError("A weighted worker may execute only one method and pinned source root.")
        return _MODULE
    kind = METHODS[method][0]
    # Reject accidental imports from a previous source before any fit occurs.
    for path in source.glob("*.py"):
        module = sys.modules.get(path.stem)
        if module is None or not getattr(module, "__file__", None):
            continue
        actual = Path(module.__file__).resolve()
        if actual != path.resolve():
            raise RuntimeError(f"Source module collision: {path.stem} already loaded from {actual}")
    sys.path.insert(0, str(source))
    entry = source / ENTRY_POINTS[kind]
    name = "_pinned_pilot_" + kind
    spec = importlib.util.spec_from_file_location(name, entry)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned entry point {entry}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run_one_trial", None)):
        raise RuntimeError(f"Pinned source has no trial entry point: {entry}")
    _BOUND, _MODULE = binding, module
    return module


def _scenario(value):
    if isinstance(value, bool):
        raise ValueError("A scenario must identify data_scenario_1 through data_scenario_4.")
    text = str(value)
    number = text.removeprefix("data_scenario_")
    if number not in ("1", "2", "3", "4"):
        raise ValueError(f"Unknown scenario: {value!r}")
    return "data_scenario_" + number, int(number)


@contextmanager
def _capture_data(module):
    import numpy as np

    original = module.generate_data
    captured = []

    def generate(*args, **kwargs):
        arrays = original(*args, **kwargs)
        if not isinstance(arrays, (tuple, list)) or len(arrays) != 4:
            raise RuntimeError("The pinned generator must return target_x, target_y, source_x, source_y.")
        canonical = [np.ascontiguousarray(array, dtype="<f8") for array in arrays]
        if not all(np.isfinite(array).all() for array in canonical):
            raise RuntimeError("Generated trial data contain non-finite values.")
        captured.append({
            "sha256": hashlib.sha256(b"".join(array.tobytes(order="C") for array in canonical)).hexdigest(),
            "shapes": [list(array.shape) for array in canonical],
        })
        return arrays

    module.generate_data = generate
    try:
        yield captured
    finally:
        module.generate_data = original


def _number(value):
    return float("nan") if value is None else float(value)


def _diagnostic_number(*candidates):
    """Keep available source diagnostics when a top-level field is missing.

    Some native trial functions retain fitted CV parameters in ESS diagnostics
    while their top-level scalar is NaN after rejecting the point estimate.
    Values are read from the existing result only; no value is reconstructed.
    """
    for candidate in candidates:
        value = _number(candidate)
        if math.isfinite(value):
            return value
    return float("nan")


def _normalize(raw, method, sid, scenario_id, source_n, gamma, seed, target_n, bootstraps, digest):
    kind, weight = METHODS[method]
    prefix = {"iwl": "iwl", "aipw": "aipwsn", "ipw": "ipw", "gcomp": "gcomp"}[kind]
    diagnostic_prefix = "aipw" if kind == "aipw" else prefix
    stem = diagnostic_prefix + "_"
    diagnostics = raw.get(stem + "ess_diagnostics", {})
    if isinstance(diagnostics, str):
        diagnostics = json.loads(diagnostics)
    diagnostics = diagnostics or {}
    full = diagnostics.get("full", {}) or {}
    estimate = _number(raw[prefix + "_estimate"])
    lower, upper = (_number(raw[prefix + suffix]) for suffix in ("_ci_lower", "_ci_upper"))
    requested = int(raw[stem + "boot_requested"])
    successful = int(raw[stem + "boot_n"])
    failed = int(raw[stem + "boot_failed"])
    if requested != bootstraps or successful + failed != requested:
        raise RuntimeError("The pinned trial returned inconsistent bootstrap accounting.")
    if int(raw["trial"]) != seed:
        raise RuntimeError("The pinned trial returned the wrong seed.")
    point_error = raw.get(stem + "point_error") or ""
    point_type = raw.get(stem + "point_error_type") or ""
    ci_error = raw.get(prefix + "_ci_error") or ""
    point_success = math.isfinite(estimate)
    ci_success = point_success and math.isfinite(lower) and math.isfinite(upper) and lower <= upper
    if ci_success and ci_error:
        raise RuntimeError("The pinned trial reports a usable CI together with a CI error.")
    result = {
        "method": method, "family": kind, "weight_model": weight,
        "seed": seed, "scenario": sid, "scenario_id": scenario_id,
        "source_n": source_n, "target_n": target_n, "gamma": gamma,
        "true_value": 1.0, "estimate": estimate, "ci_lower": lower, "ci_upper": upper,
        "interval_type": "bootstrap_percentile_95", "point_success": point_success,
        "ci_success": ci_success, "boot_requested": requested, "boot_success": successful,
        "boot_failed": failed, "boot_failure_rate": _number(raw.get(stem + "boot_failure_rate")),
        "boot_se": _number(raw.get(prefix + "_boot_se")), "point_error": str(point_error),
        "point_error_type": str(point_type), "ci_error": str(ci_error),
        "ess": _diagnostic_number(raw.get(stem + "ess"), full.get("ess")),
        "positive_weights": _diagnostic_number(raw.get(stem + "positive_weights"), full.get("positive_weights")),
        "max_normalized_weight": _diagnostic_number(raw.get(stem + "max_normalized_weight"), full.get("max_normalized_weight")),
        "propensity_C": _diagnostic_number(raw.get(stem + "propensity_C"), diagnostics.get("propensity_C")),
        "ratio_sigma": _diagnostic_number(raw.get(stem + "ratio_sigma"), diagnostics.get("ratio_sigma")),
        "ratio_lambda": _diagnostic_number(raw.get(stem + "ratio_lambda"), diagnostics.get("ratio_lambda")),
        "l1_penalty": 0.0 if kind == "iwl" else None,
        "data_sha256": digest,
    }
    return result


def run_trial(method, scenario, source_n, gamma, seed, context):
    """Run one genuine repository trial; the caller owns checkpoints and files.

    ``context`` names pinned source roots and the actual bootstrap count. A prep
    profile may use fewer replicates, but the final study context specifies 200.
    No RNG is touched before the repository itself seeds and generates its data.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown frequentist method: {method!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("The trial seed must be a nonnegative integer.")
    if isinstance(source_n, bool) or not isinstance(source_n, int) or source_n < 3:
        raise ValueError("source_n must be an integer of at least three.")
    target_n = context["target_n"]
    bootstraps = context["bootstrap_replicates"]
    if isinstance(target_n, bool) or not isinstance(target_n, int) or target_n < 3:
        raise ValueError("target_n must be an integer of at least three.")
    if isinstance(bootstraps, bool) or not isinstance(bootstraps, int) or bootstraps < 2:
        raise ValueError("At least two bootstrap replicates are required.")
    gamma = float(gamma)
    if not math.isfinite(gamma):
        raise ValueError("Gamma must be finite.")
    sid, scenario_id = _scenario(scenario)
    kind, weight = METHODS[method]
    if kind == "gcomp" and scenario_id != 4:
        raise ValueError("The requested G-computation design contains only scenario 4.")
    source = Path(context["sources"][kind]).resolve(strict=True)
    module = _load_source(method, source)
    minimum = float(context.get("min_ess", 3.0))
    maximum = float(context.get("max_bootstrap_failure_rate", .05))
    original_learner = getattr(module, "ImportaceWeightLearn", None)
    try:
        with _capture_data(module) as captured:
            if kind == "iwl":
                if module.SPLIT_RATIO != 1.0 or module.IWL_ALPHA != 1.0:
                    raise RuntimeError("Pinned IWL must use the full target sample and alpha=1.")
                options = dict(lambda_selection="fixed", lambda_=0., min_ess=minimum)
                if weight == "oracle":
                    from oracle_weighted_learn import OracleWeightedLearn
                    config = dict(scenario={"data_scenario_id": sid}, hyperparameters={"n_features": 2})
                    module.ImportaceWeightLearn = functools.partial(
                        OracleWeightedLearn, config=config, config_dir=source / "configs")
                else:
                    options.update(weight_model=weight, ratio_nonnegative="prediction", ratio_cv=True)
                    if weight == "ps_logistic":
                        options.update(propensity_penalty="l2", propensity_cv=True,
                                       propensity_C=1., propensity_C_grid=None, propensity_cv_folds=5)
                raw = module.run_one_trial(seed, sid, source_n, gamma, bootstraps, target_n,
                                           weight_kwargs=options, max_bootstrap_failure_rate=maximum)
            elif kind == "aipw":
                options = dict(weight_model=weight, min_ess=minimum, ratio_nonnegative="prediction",
                               ratio_cv=True, propensity_cv=True, propensity_C=1.,
                               propensity_C_grid=None, propensity_cv_folds=5, propensity_clip=None)
                raw = module.run_one_trial(seed, sid, source_n, gamma, bootstraps, target_n,
                                           options, max_bootstrap_failure_rate=maximum)
            elif kind == "ipw":
                options = dict(min_ess=minimum, propensity_penalty="l2", propensity_cv=True,
                               propensity_C=1., propensity_C_grid=None, propensity_cv_folds=5)
                raw = module.run_one_trial(seed, sid, source_n, gamma, bootstraps, target_n,
                                           weight, options, max_bootstrap_failure_rate=maximum)
            else:
                raw = module.run_one_trial(seed, sid, source_n, gamma, bootstraps, target_n,
                                           max_failure_rate=maximum)
    finally:
        if original_learner is not None:
            module.ImportaceWeightLearn = original_learner
    if len(captured) != 1:
        raise RuntimeError(f"Expected one actual data generation; observed {len(captured)}.")
    digest = captured[0]["sha256"]
    if "data_sha256" in raw and raw["data_sha256"] != digest:
        raise RuntimeError("The pinned source and adapter disagree on the trial data digest.")
    normalized = _normalize(raw, method, sid, scenario_id, source_n, gamma, seed,
                            target_n, bootstraps, digest)
    return {"raw": raw, "data_sha256": digest, "data_shapes": captured[0]["shapes"],
            "normalized": [normalized]}
