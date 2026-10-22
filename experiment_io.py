"""Prevent mixing runs with different estimators, ESS rules or source code."""

import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import pandas as pd


def _atomic_text(path, text):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_run_meta(out_dir, root, **meta):
    out_dir, root = Path(out_dir), Path(root)
    meta.update(bootstrap_ess_threshold_applied=False,
                bootstrap_min_positive_weights=3, ess_scope="original_sample_only")
    code = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(
            list(root.glob("*.py"))
            + list((root / "scripts").glob("nonlinear_gamma_*.py"))
        )
    }
    versions = {
        key: importlib.metadata.version(key)
        for key in ("numpy", "scipy", "scikit-learn", "pandas")
    }
    contract = {
        key: value
        for key, value in meta.items()
        if key not in ("n_trials", "scenarios", "source_ns", "gammas")
    }
    contract.update(
        ess_contract_version=2,
        source_sha256=code,
        versions=versions,
        python=platform.python_version(),
    )
    # Normalize tuples and other JSON-serializable metadata consistently.
    contract = json.loads(json.dumps(contract, default=str))
    path = out_dir / "run_meta.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("resume_contract") != contract:
            raise ValueError(
                "ESS / Density-ratio settings or code differ from saved results. Use a new output directory."
            )
    elif any(out_dir.glob("trials_*.csv")) or any(
        (out_dir / "checkpoints").rglob("trial_*.json")
    ):
        raise ValueError(
            "Existing trials lack metadata for ESS. Use a new output directory."
        )
    meta.update(
        resume_contract=contract,
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        host=platform.node(),
    )
    _atomic_text(path, json.dumps(meta, indent=2, ensure_ascii=False, default=str))


def load_or_run_trials(csv_path, n_trials, required_col, run_fn, n_jobs, verbose):
    """Persist each completed trial, including failed estimates; never discard old rows."""
    from joblib import Parallel, delayed

    csv_path = Path(csv_path)
    checkpoint = csv_path.parent / "checkpoints" / csv_path.stem
    checkpoint.mkdir(parents=True, exist_ok=True)
    rows = {}

    def add(row):
        trial = row.get("trial")
        if (
            not isinstance(trial, (int, np.integer))
            or trial < 0
            or required_col not in row
        ):
            raise ValueError("Invalid trial checkpoint schema.")
        rows[int(trial)] = {
            key: np.nan if value is None else value for key, value in row.items()
        }

    if csv_path.exists():
        cached = pd.read_csv(csv_path)
        if (
            "trial" not in cached
            or required_col not in cached
            or not any(c.endswith("boot_failed_ESS") for c in cached)
        ):
            raise ValueError(
                "Existing trials lack ESS diagnostics. Use a new output directory."
            )
        if (
            cached.trial.duplicated().any()
            or not np.isfinite(cached.trial).all()
            or (cached.trial < 0).any()
            or (cached.trial % 1 != 0).any()
        ):
            raise ValueError("Invalid or duplicate cached trial IDs.")
        for row in cached.to_dict("records"):
            row["trial"] = int(row["trial"])
            add(row)
    for path in sorted(checkpoint.glob("trial_*.json")):
        add(json.loads(path.read_text(encoding="utf-8")))
    todo = [i for i in range(n_trials) if i not in rows]

    def run_and_save(i):
        row = run_fn(i)
        if row.get("trial") != i or required_col not in row:
            raise ValueError("Invalid computed trial schema.")
        # Preserve full float precision and use JSON null for missing estimates.
        payload = json.dumps(_json_value(row), ensure_ascii=False, allow_nan=False)
        _atomic_text(checkpoint / f"trial_{i:06d}.json", payload)
        return row

    for row in Parallel(
        n_jobs=n_jobs, backend="loky", verbose=verbose, return_as="generator_unordered"
    )(delayed(run_and_save)(i) for i in todo):
        add(row)
    frame = (
        pd.DataFrame(list(rows.values())).sort_values("trial").reset_index(drop=True)
    )
    _atomic_text(csv_path, frame.to_csv(index=False))
    return frame[frame.trial < n_trials].reset_index(drop=True), len(todo)
