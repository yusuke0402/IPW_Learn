"""Thin launch adapters for the pinned Conditional and Commensurate PP pilots.

The caller must use an isolated worker process for one method/source tree and
set its numerical thread limits before importing scientific Python. No posterior
or data-generating equations are implemented here: the source repositories'
existing single-trial functions do all statistical work.

``context['mode']`` defaults to ``production``. Production MCMC/prior settings
must equal the pinned source configuration. ``verification`` permits bounded
MCMC iteration counts; all production priors and sampler defaults remain pinned.
"""

from contextlib import contextmanager
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys


METHODS = ("conditional_pp", "commensurate_pp")
_BOUND = None
_MODULES = {}


def _scenario(value):
    if value == 4 or value == "4" or value == "data_scenario_4":
        return "data_scenario_4"
    raise ValueError("The Bayesian pilot is restricted to data_scenario_4.")


def _validate_request(method, scenario, source_n, gamma, seed, context):
    if method not in METHODS:
        raise ValueError(f"Unknown Bayesian method: {method!r}")
    scenario = _scenario(scenario)
    if isinstance(source_n, bool) or source_n not in (20, 100, 500):
        raise ValueError("source_n must be 20, 100, or 500.")
    if not math.isfinite(float(gamma)) or float(gamma) not in (0., .5):
        raise ValueError("gamma must be 0 or 0.5.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 200:
        raise ValueError("The pilot seed must be an integer from 0 through 199.")
    if context.get("target_n", 50) != 50:
        raise ValueError("The pilot target sample size is fixed at 50.")
    mode = context.get("mode", "production")
    if mode not in ("production", "verification"):
        raise ValueError("mode must be production or verification.")
    return scenario, int(source_n), float(gamma), seed, mode


@contextmanager
def _source_environment(source, module_names):
    """Give relative config reads and absolute imports the correct source tree."""
    source = Path(source).resolve(strict=True)
    for name in module_names:
        module = sys.modules.get(name)
        if module is None:
            continue
        filename = getattr(module, "__file__", None)
        if filename is None or not Path(filename).resolve().is_relative_to(source):
            raise RuntimeError(
                f"Module {name!r} was loaded outside {source}; use an isolated worker."
            )
    previous_cwd, previous_path = Path.cwd(), list(sys.path)
    sys.path.insert(0, str(source))
    os.chdir(source)
    try:
        yield source
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path


def _module(source, relative, label):
    path = Path(source) / relative
    key = str(path.resolve())
    if key not in _MODULES:
        name = "_pilot_bayes_" + label + "_" + hashlib.sha256(key.encode()).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        _MODULES[key] = module
    return _MODULES[key]


def _data_observer(source_n, target_n, expected_digest=None):
    """Observe the arrays returned to the real fitter without touching its RNG."""
    observed = {}

    def observe(arrays):
        import numpy as np

        if observed:
            raise RuntimeError("A single trial unexpectedly generated data more than once.")
        if len(arrays) != 4:
            raise ValueError("The fitter data must contain target X/y and source X/y.")
        tx, ty, sx, sy = (np.asarray(array) for array in arrays)
        if tx.shape != (target_n, 3) or sx.shape != (source_n, 3):
            raise ValueError(f"Unexpected Bayesian design shapes: {tx.shape}, {sx.shape}")
        if ty.shape != (target_n, 1) or sy.shape != (source_n, 1):
            raise ValueError(f"Unexpected Bayesian outcome shapes: {ty.shape}, {sy.shape}")
        if not all(np.isfinite(array).all() for array in (tx, ty, sx, sy)):
            raise ValueError("The Bayesian data arrays must be finite.")
        if not np.all(tx[:, 0] == 1.) or not np.all(sx[:, 0] == 1.):
            raise ValueError("The Bayesian design matrices must retain their intercept column.")
        digest = hashlib.sha256(b"".join(
            np.asarray(array, dtype="<f8", order="C").tobytes(order="C")
            for array in (tx, ty, sx, sy)
        )).hexdigest()
        if expected_digest is not None and digest != expected_digest:
            raise ValueError("Actual Bayesian fitting data differ from the paired-data contract.")
        observed.update(data_sha256=digest, shapes=[list(array.shape) for array in (tx, ty, sx, sy)])
        return arrays

    return observed, observe


@contextmanager
def _observe_function(module, name, observe):
    original = getattr(module, name)

    def wrapped(*args, **kwargs):
        return observe(original(*args, **kwargs))

    setattr(module, name, wrapped)
    try:
        yield
    finally:
        setattr(module, name, original)


def _settings(defaults, overrides, mode, verification_keys=()):
    result = dict(defaults)
    overrides = {} if overrides is None else dict(overrides)
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown source configuration settings: {sorted(unknown)}")
    for name, value in overrides.items():
        if value != defaults[name] and not (mode == "verification" and name in verification_keys):
            raise ValueError(f"The pinned {name} setting may not change in {mode} mode.")
        result[name] = value
    return result


def _base_row(method, scenario, source_n, gamma, seed, digest):
    return dict(
        method=method, seed=seed, scenario=scenario, scenario_id=4,
        source_n=source_n, target_n=50, gamma=gamma, true_value=1.,
        interval_type="posterior", ci_level=.95, data_sha256=digest,
        point_success=True, ci_success=True,
        boot_requested=None, boot_success=None, boot_failed=None,
        boot_se=None, boot_failure_rate=None, point_error="", point_error_type="",
        ci_error="", ess=None, positive_weights=None, max_normalized_weight=None,
    )


def _check_output(row):
    for key in ("estimate", "ci_lower", "ci_upper"):
        if not math.isfinite(float(row[key])):
            raise ValueError(f"Bayesian {key} was not finite.")
    if row["ci_lower"] > row["ci_upper"]:
        raise ValueError("The Bayesian posterior interval is reversed.")


def run_trial(method, scenario, source_n, gamma, seed, context):
    """Execute exactly one repository-native Bayesian trial and normalize its row.

    The result contains the untouched raw trial result, normalized scalar rows,
    observed data digest/shapes, and effective inference settings. Statistical
    failures propagate to the coordinator rather than becoming invented output.
    """
    import yaml

    scenario, source_n, gamma, seed, mode = _validate_request(
        method, scenario, source_n, gamma, seed, context
    )
    source_key = {"conditional_pp": "conditional", "commensurate_pp": "commensurate"}[method]
    source = Path(context["sources"][source_key]).resolve(strict=True)
    global _BOUND
    binding = (method, str(source))
    if _BOUND is not None and _BOUND != binding:
        raise RuntimeError("A Bayesian worker may execute only one method and pinned source root.")
    _BOUND = binding
    observed, observe = _data_observer(source_n, 50, context.get("expected_data_sha256"))
    normalized = []

    if method == "conditional_pp":
        with _source_environment(source, ("power_prior",)):
            module = _module(source, "scripts/nonlinear_gamma_pp.py", "conditional")
            configured_a0 = tuple(context.get("conditional_pp_a0", (.1, .25, .5, .75, 1.)))
            if configured_a0 != (.1, .25, .5, .75, 1.) or tuple(module.A0S) != configured_a0:
                raise ValueError("Conditional PP must retain the requested five a0 values.")
            previous_target_n = module.TARGET_N
            module.TARGET_N = 50
            try:
                with _observe_function(module, "generate_data", observe):
                    raw = module.run_one_trial(seed, scenario, source_n, gamma)
            finally:
                module.TARGET_N = previous_target_n
            for a0 in configured_a0:
                prefix = f"pp{a0:g}".replace(".", "")
                label = "conditional_pp_a0_" + f"{a0:g}".replace(".", "p")
                row = _base_row(label, scenario, source_n, gamma, seed, observed["data_sha256"])
                row.update(estimate=float(raw[prefix + "_estimate"]),
                           ci_lower=float(raw[prefix + "_ci_lower"]),
                           ci_upper=float(raw[prefix + "_ci_upper"]),
                           a0=float(a0), estimate_statistic="posterior_mean")
                normalized.append(row)
            settings = dict(mode=mode, target_n=50, a0_values=list(configured_a0), ci_level=.95)

    elif method == "commensurate_pp":
        with _source_environment(source, ("data", "cpp")):
            module = _module(source, "simulation.py", "commensurate")
            config = yaml.safe_load((source / "configs/config.yaml").read_text())
            mc = _settings(config["mcmc"], context.get("commensurate_mcmc"), mode,
                           verification_keys=("n_iter", "burn_in", "thin"))
            if mc["seed"] != 0:
                raise ValueError("CPP's base data seed must remain zero for pairing by trial seed.")
            config["scenario"].update(data_scenario_id=scenario, gamma_x1_sq=gamma)
            config["hyperparameters"].update(n_features=2, split_ratio=1., n_trial=1, n_jobs=1)
            config["dataset"].update(true_value=1., target_number=50, source_number=source_n)
            config["mcmc"] = mc
            with _observe_function(module, "_target_source", observe):
                raw = module.run_one_trial(seed, config)
            row = _base_row(method, scenario, source_n, gamma, seed, observed["data_sha256"])
            row.update(estimate=float(raw["ate_median"]), ci_lower=float(raw["ci_low"]),
                       ci_upper=float(raw["ci_high"]), estimate_statistic="posterior_median")
            for name in ("alpha0_mean", "tau_median", "acc_log_tau", "acc_alpha0"):
                row[name] = float(raw[name])
            normalized.append(row)
            settings = dict(mode=mode, target_n=50, data_seed_policy="base_seed_zero_plus_trial",
                            posterior_seed_policy="10000_plus_trial", mcmc=mc, ci_level=.95)

    if len(observed) == 0:
        raise RuntimeError("No actual fitting data were observed.")
    if raw.get("trial") != seed:
        raise ValueError("The repository trial result does not retain the requested seed.")
    for row in normalized:
        _check_output(row)
    return dict(raw=raw, data_sha256=observed["data_sha256"],
                data_shapes=observed["shapes"], normalized=normalized, effective_settings=settings)
