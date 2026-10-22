"""Aggregate completed pilot checkpoints without treating missing results as zero.

This module needs only Python's standard library. Plotting is optional and uses
matplotlib. The coordinator owns checkpoint creation and source/run validation;
this reader independently checks identities, fingerprints and paired data hashes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


TRUE_VALUE = 1.0
IDENTITY = ("method", "scenario", "gamma", "target_n", "source_n", "seed")
CONDITION = IDENTITY[:-1] + ("interval_type",)
TRIAL_COLUMNS = [
    "method", "method_label", "scenario", "gamma", "target_n", "source_n", "seed",
    "estimate", "estimate_valid", "ci_lower", "ci_upper", "ci_valid", "interval_type",
    "point_error", "ci_error", "boot_requested", "boot_success", "boot_failed",
    "ess", "rhat_max", "n_divergent", "data_sha256", "fingerprint", "checkpoint",
]
SUMMARY_COLUMNS = [
    "method", "method_label", "scenario", "gamma", "target_n", "source_n", "interval_type",
    "n_total", "n_estimate_success", "n_estimate_failed", "n_ci_success", "n_ci_failed",
    "mean", "bias", "sd", "rmse", "coverage", "coverage_denominator", "mean_ci_width",
    "bootstrap_requested", "bootstrap_success", "bootstrap_failed", "bootstrap_failure_rate",
    "ess_median", "rhat_max", "n_rhat_available", "n_rhat_gt_1p05", "n_divergent_total",
    "n_divergence_diagnostics_available", "n_trials_with_divergence", "run_mode",
]
PAIRED_COLUMNS = [
    "method", "method_label", "reference_method", "scenario", "gamma", "target_n", "source_n",
    "n_seed_overlap", "n_pairs", "n_pairs_missing_estimate", "mean_estimate_difference",
    "sd_estimate_difference", "rmse_estimate_difference", "run_mode",
]


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(value, label):
    if value is None or value == "":
        return math.nan
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    return result if math.isfinite(result) else math.nan


def _integer(value, label, minimum=0):
    number = _number(value, label)
    if not _finite(number) or not number.is_integer() or number < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(number)


def _scenario(value):
    if isinstance(value, str) and value.startswith("data_scenario_"):
        value = value.removeprefix("data_scenario_")
    return _integer(value, "scenario", 1)


def _sha(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA256 hex digest")
    return value


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _csv_value(value):
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return "NA"
    if isinstance(value, (dict, tuple, list)):
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, allow_nan=False)
    return value


def _write_csv(path, columns, rows):
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _csv_value(row.get(name)) for name in columns})
    _atomic_text(path, stream.getvalue())


def _method_label(method):
    labels = {
        "iwl_ulsif_l1none": "IWL / uLSIF / L1=0", "iwl_oracle_l1none": "IWL / oracle / L1=0",
        "iwl_logistic_l1none": "IWL / logistic / L1=0", "aipw_sn_ulsif": "AIPW-SN / uLSIF",
        "aipw_sn_oracle": "AIPW-SN / oracle", "aipw_sn_logistic": "AIPW-SN / logistic",
        "ipw_sn_ulsif": "IPW-SN / uLSIF", "ipw_sn_oracle": "IPW-SN / oracle",
        "ipw_sn_logistic": "IPW-SN / logistic", "g_computation": "G-computation",
        "commensurate_pp": "Commensurate PP",
    }
    if method.startswith("conditional_pp_a0_"):
        return "Conditional PP / a0=" + method.removeprefix("conditional_pp_a0_").replace("p", ".")
    return labels.get(method, method)


def _manifest(output_dir):
    for name in ("manifest.json", "execution_manifest.json", "run_manifest.json", "configs/manifest.json", "configs/execution_manifest.json"):
        path = output_dir / name
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"Run manifest must be an object: {path}")
            return value, str(path.relative_to(output_dir))
    return {}, None


def _mode(manifest, output_dir):
    for container in (manifest, manifest.get("run", {}), manifest.get("settings", {})):
        if isinstance(container, dict):
            for key in ("mode", "run_mode"):
                if container.get(key):
                    return str(container[key])
    return "verification" if "verification" in output_dir.parts else "unclassified"


def _normalize(item, payload, path, output_dir):
    if not isinstance(item, dict):
        raise ValueError(f"Normalized rows must be objects: {path}")
    job = payload["job"]
    row = dict(item)
    for name in ("method", "scenario", "source_n", "target_n", "gamma", "seed"):
        if name not in row:
            row[name] = job.get(name)
    if not isinstance(row["method"], str) or not row["method"]:
        raise ValueError(f"Normalized method is missing: {path}")
    row["scenario"] = _scenario(row["scenario"])
    for name in ("source_n", "target_n"):
        row[name] = _integer(row[name], name, 1)
    row["seed"] = _integer(row["seed"], "seed")
    row["gamma"] = _number(row["gamma"], "gamma")
    if not _finite(row["gamma"]):
        raise ValueError(f"Gamma must be finite: {path}")
    for name in ("estimate", "ci_lower", "ci_upper", "ess", "boot_requested", "boot_success", "boot_failed"):
        row[name] = _number(row.get(name), name)
    for name in ("boot_requested", "boot_success", "boot_failed"):
        if _finite(row[name]) and (row[name] < 0 or not row[name].is_integer()):
            raise ValueError(f"{name} must be a nonnegative integer: {path}")
    if all(_finite(row[name]) for name in ("boot_requested", "boot_success", "boot_failed")):
        if row["boot_success"] + row["boot_failed"] != row["boot_requested"]:
            raise ValueError(f"Bootstrap counts do not add up: {path}")
    row["point_error"] = row.get("point_error") or ""
    row["ci_error"] = row.get("ci_error") or ""
    row["estimate_valid"] = _finite(row["estimate"]) and not bool(row["point_error"])
    row["ci_valid"] = (row["estimate_valid"] and not bool(row["ci_error"])
                       and _finite(row["ci_lower"]) and _finite(row["ci_upper"]))
    if row["ci_valid"] and row["ci_lower"] > row["ci_upper"]:
        raise ValueError(f"Lower confidence/credible bound exceeds upper bound: {path}")
    row["interval_type"] = row.get("interval_type") or "unclassified"
    row["method_label"] = _method_label(row["method"])
    row["rhat_max"] = _number(row.get("rhat_max", row.get("rhat")), "rhat_max")
    row["n_divergent"] = _number(row.get("n_divergent", row.get("divergences")), "n_divergent")
    if _finite(row["n_divergent"]) and (row["n_divergent"] < 0 or not row["n_divergent"].is_integer()):
        raise ValueError(f"Divergence count must be a nonnegative integer: {path}")
    row["data_sha256"] = _sha(payload.get("data_sha256"), "data_sha256")
    row["fingerprint"] = _sha(payload.get("fingerprint"), "fingerprint")
    row["checkpoint"] = str(path.relative_to(output_dir))
    # A row must describe the actual checkpoint's data and seed, including CondPP variants.
    for name, convert in (("scenario", _scenario), ("source_n", lambda x: _integer(x, "source_n", 1)),
                          ("gamma", lambda x: _number(x, "gamma")), ("seed", lambda x: _integer(x, "seed"))):
        if name in job and row[name] != convert(job[name]):
            raise ValueError(f"Normalized {name} does not match checkpoint job: {path}")
    if "target_n" in job and row["target_n"] != _integer(job["target_n"], "target_n", 1):
        raise ValueError(f"Normalized target_n does not match checkpoint job: {path}")
    return row


def _mean(values):
    return statistics.fmean(values) if values else math.nan


def _sd(values):
    return statistics.stdev(values) if len(values) >= 2 else math.nan


def _summary(rows, mode):
    result = []
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in CONDITION)].append(row)
    for key, group in sorted(groups.items()):
        estimates = [row["estimate"] for row in group if row["estimate_valid"]]
        intervals = [row for row in group if row["ci_valid"]]
        requested = [row["boot_requested"] for row in group if _finite(row["boot_requested"])]
        successful = [row["boot_success"] for row in group if _finite(row["boot_success"])]
        failed = [row["boot_failed"] for row in group if _finite(row["boot_failed"])]
        ess = [row["ess"] for row in group if _finite(row["ess"])]
        rhats = [row["rhat_max"] for row in group if _finite(row["rhat_max"])]
        divergences = [row["n_divergent"] for row in group if _finite(row["n_divergent"])]
        n_boot = sum(requested) if requested else math.nan
        n_failed = sum(failed) if failed else math.nan
        value = dict(zip(CONDITION, key))
        value.update(method_label=_method_label(value["method"]), n_total=len(group),
            n_estimate_success=len(estimates), n_estimate_failed=len(group)-len(estimates),
            n_ci_success=len(intervals), n_ci_failed=len(group)-len(intervals),
            mean=_mean(estimates), bias=_mean([x-TRUE_VALUE for x in estimates]), sd=_sd(estimates),
            rmse=math.sqrt(_mean([(x-TRUE_VALUE)**2 for x in estimates])) if estimates else math.nan,
            coverage=_mean([float(row["ci_lower"] <= TRUE_VALUE <= row["ci_upper"]) for row in intervals]),
            coverage_denominator=len(intervals),
            mean_ci_width=_mean([row["ci_upper"]-row["ci_lower"] for row in intervals]),
            bootstrap_requested=n_boot, bootstrap_success=sum(successful) if successful else math.nan,
            bootstrap_failed=n_failed,
            bootstrap_failure_rate=n_failed/n_boot if _finite(n_boot) and n_boot > 0 and _finite(n_failed) else math.nan,
            ess_median=statistics.median(ess) if ess else math.nan,
            rhat_max=max(rhats) if rhats else math.nan, n_rhat_available=len(rhats),
            n_rhat_gt_1p05=sum(x > 1.05 for x in rhats),
            n_divergent_total=sum(divergences) if divergences else math.nan,
            n_divergence_diagnostics_available=len(divergences),
            n_trials_with_divergence=sum(x > 0 for x in divergences), run_mode=mode)
        result.append(value)
    return result


def _paired(rows, mode):
    groups = defaultdict(dict)
    for row in rows:
        if row["scenario"] == 4:
            key = (row["method"], row["scenario"], row["gamma"], row["target_n"], row["source_n"])
            groups[key][row["seed"]] = row
    result = []
    for key, group in sorted(groups.items()):
        method, scenario, gamma, target_n, source_n = key
        if method == "g_computation":
            continue
        reference = groups.get(("g_computation", scenario, gamma, target_n, source_n))
        if reference is None:
            continue
        seeds = sorted(set(group).intersection(reference))
        differences = [group[s]["estimate"]-reference[s]["estimate"] for s in seeds
                       if group[s]["estimate_valid"] and reference[s]["estimate_valid"]]
        result.append(dict(method=method, method_label=_method_label(method), reference_method="g_computation",
            scenario=scenario, gamma=gamma, target_n=target_n, source_n=source_n,
            n_seed_overlap=len(seeds), n_pairs=len(differences), n_pairs_missing_estimate=len(seeds)-len(differences),
            mean_estimate_difference=_mean(differences), sd_estimate_difference=_sd(differences),
            rmse_estimate_difference=math.sqrt(_mean([x*x for x in differences])) if differences else math.nan,
            run_mode=mode))
    return result


def build_report(output_dir, expected_jobs=None):
    """Validate and aggregate completed checkpoints; return a progress/status dict.

    ``expected_jobs`` may be an integer count or the coordinator's job sequence.
    No trials are fabricated for pending jobs. CSV ``NA`` denotes missing values.
    """
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise ValueError(f"Output directory does not exist: {output_dir}")
    manifest, manifest_path = _manifest(output_dir)
    mode = _mode(manifest, output_dir)
    if expected_jobs is None:
        expected_jobs = manifest.get("expected_jobs", manifest.get("job_count", manifest.get("jobs")))
    if expected_jobs is None:
        expected_count = None
    elif isinstance(expected_jobs, Sequence) and not isinstance(expected_jobs, (str, bytes)):
        expected_count = len(expected_jobs)
    else:
        expected_count = _integer(expected_jobs, "expected_jobs")
    expected_fingerprint = manifest.get("fingerprint", manifest.get("run_fingerprint"))
    if expected_fingerprint is not None:
        expected_fingerprint = _sha(expected_fingerprint, "manifest fingerprint")
    rows, ignored, completed_paths = [], [], []
    row_identities, job_identities, fingerprints, paired_hashes = set(), set(), set(), {}
    for path in sorted((output_dir / "raw").rglob("trial_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Checkpoint must be an object: {path}")
        if "result_sha256" in payload:
            expected_checksum = _sha(payload["result_sha256"], "result_sha256")
            checksum_payload = {key: value for key, value in payload.items() if key != "result_sha256"}
            serialized = json.dumps(checksum_payload, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False)
            if hashlib.sha256(serialized.encode()).hexdigest() != expected_checksum:
                raise ValueError(f"Checkpoint content checksum mismatch: {path}")
        if payload.get("status") in {"failed", "error", "running", "pending", "started"}:
            ignored.append(dict(checkpoint=str(path.relative_to(output_dir)), status=payload["status"]))
            continue
        if not isinstance(payload.get("job"), dict) or not isinstance(payload.get("normalized"), list) or not payload["normalized"]:
            raise ValueError(f"Completed checkpoint has no job/normalized rows: {path}")
        normalized = [_normalize(item, payload, path, output_dir) for item in payload["normalized"]]
        fingerprint = normalized[0]["fingerprint"]
        fingerprints.add(fingerprint)
        if len(fingerprints) > 1 or expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise ValueError(f"Checkpoint fingerprint does not match this run: {path}")
        first = normalized[0]
        job_identity = (payload["job"].get("method"),) + tuple(first[name] for name in IDENTITY[1:])
        if job_identity in job_identities:
            raise ValueError(f"Duplicate job checkpoint: {path}")
        job_identities.add(job_identity)
        for row in normalized:
            identity = tuple(row[name] for name in IDENTITY)
            if identity in row_identities:
                raise ValueError(f"Duplicate normalized method/condition/seed: {path}")
            row_identities.add(identity)
            data_identity = tuple(row[name] for name in IDENTITY[1:])
            previous_hash = paired_hashes.setdefault(data_identity, row["data_sha256"])
            if previous_hash != row["data_sha256"]:
                raise ValueError(f"Methods received different data for the same condition and seed: {path}")
            rows.append(row)
        completed_paths.append(str(path.relative_to(output_dir)))
    if expected_count is not None and len(completed_paths) > expected_count:
        raise ValueError("Completed job count exceeds the declared run job count")
    rows.sort(key=lambda row: tuple(row[name] for name in IDENTITY))
    summaries, paired = _summary(rows, mode), _paired(rows, mode)
    extra_columns = sorted({key for row in rows for key in row} - set(TRIAL_COLUMNS))
    summary_dir = output_dir / "summary"
    _write_csv(summary_dir / "normalized_trials.csv", TRIAL_COLUMNS + extra_columns, rows)
    _write_csv(summary_dir / "per_condition_summary.csv", SUMMARY_COLUMNS, summaries)
    # A concise stable name is also provided for downstream comparisons.
    _write_csv(summary_dir / "summary.csv", SUMMARY_COLUMNS, summaries)
    _write_csv(summary_dir / "paired_vs_gcomp.csv", PAIRED_COLUMNS, paired)
    status = dict(schema_version=1, updated_at_utc=datetime.now(timezone.utc).isoformat(),
        run_mode=mode, manifest=manifest_path,
        fingerprint=next(iter(fingerprints), expected_fingerprint),
        expected_jobs=expected_count, completed_jobs=len(completed_paths),
        remaining_jobs=expected_count-len(completed_paths) if expected_count is not None else None,
        ignored_noncompleted_checkpoints=ignored, normalized_trial_rows=len(rows), condition_rows=len(summaries),
        valid_estimates=sum(row["estimate_valid"] for row in rows), valid_intervals=sum(row["ci_valid"] for row in rows),
        true_value=TRUE_VALUE, missing_value="NA",
        coverage_rule="Among finite ordered intervals with a usable point estimate and no recorded CI error; denominator is explicit.",
        paired_rule="Scenario 4, matching condition and seed, matching observed data SHA256, both point estimates usable.",
        interpretation=("Startup verification only; these reduced samples are not a pilot accuracy comparison."
                        if mode in {"verification", "preflight", "smoke"} else
                        "Partial results until every declared job is completed; Monte Carlo precision depends on usable counts."),
        files=["summary/normalized_trials.csv", "summary/per_condition_summary.csv", "summary/summary.csv", "summary/paired_vs_gcomp.csv"])
    _atomic_text(summary_dir / "report_status.json", json.dumps(_json_safe(status), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return status


def build_plots(output_dir):
    """Generate scientific metric plots from the already validated summary CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir).resolve()
    status = json.loads((output_dir / "summary/report_status.json").read_text())
    with (output_dir / "summary/per_condition_summary.csv").open(newline="") as handle:
        summaries = list(csv.DictReader(handle))
    groups = defaultdict(list)
    for row in summaries:
        groups[(int(row["scenario"]), float(row["gamma"]), int(row["target_n"]))].append(row)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    metrics = [("bias", "Bias"), ("rmse", "RMSE"), ("coverage", "95% interval coverage"), ("mean_ci_width", "Mean interval width")]
    for (scenario, gamma, target_n), group in sorted(groups.items()):
        if not any(_finite(_number(row["mean"] if row["mean"] != "NA" else None, "mean")) for row in group):
            continue
        methods = sorted({row["method"] for row in group})
        figure, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
        handles, labels = [], []
        colors = plt.get_cmap("tab20")
        for index, method in enumerate(methods):
            selected = sorted((row for row in group if row["method"] == method), key=lambda row: int(row["source_n"]))
            for axis, (key, label) in zip(axes.ravel(), metrics):
                x = [int(row["source_n"]) for row in selected]
                y = [_number(row[key] if row[key] != "NA" else None, key) for row in selected]
                line, = axis.plot(x, y, marker="o", markersize=4, color=colors(index % 20), label=_method_label(method))
                axis.set(xlabel="Source sample size", ylabel=label, xticks=sorted({int(row["source_n"]) for row in group}))
                axis.grid(alpha=.2)
                if key == "bias":
                    axis.axhline(0., color="black", lw=.7, alpha=.6)
                elif key == "coverage":
                    axis.axhline(.95, color="black", ls="--", lw=.7)
                    axis.set_ylim(-.03, 1.03)
                if key == "bias":
                    handles.append(line)
                    labels.append(_method_label(method))
        prefix = "Verification only" if status["run_mode"] in {"verification", "preflight", "smoke"} else "Pilot"
        if status["remaining_jobs"]:
            prefix += " (partial)"
        figure.suptitle(f"{prefix}: scenario {scenario}, gamma={gamma:g}, target={target_n}\n"
                       f"Completed jobs: {status['completed_jobs']} / {status['expected_jobs'] or '?'}; valid-CI denominators are in the CSV")
        figure.legend(handles, labels, loc="outside lower center", ncol=min(4, len(labels)), fontsize=8)
        # Keep only the gamma token escaped, leaving the PNG suffix unchanged.
        name = f"pilot_scenario_{scenario:02d}_gamma_{format(gamma, 'g').replace('.', 'p')}_T{target_n:03d}_metrics.png"
        descriptor, temporary = tempfile.mkstemp(prefix=".pilot_plot_", suffix=".png", dir=figure_dir)
        os.close(descriptor)
        try:
            figure.savefig(temporary, dpi=150)
            os.replace(temporary, figure_dir / name)
        finally:
            plt.close(figure)
            if os.path.exists(temporary):
                os.unlink(temporary)
        generated.append(str((figure_dir / name).relative_to(output_dir)))
    # Delete only this generator's obsolete derived plots, never arbitrary figures.
    for previous in figure_dir.glob("pilot_scenario_*_metrics.png"):
        if str(previous.relative_to(output_dir)) not in generated:
            previous.unlink()
    status["figures"] = generated
    _atomic_text(output_dir / "summary/report_status.json", json.dumps(_json_safe(status), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-jobs", type=int)
    parser.add_argument("--plots", action="store_true", help="Also generate metric PNGs using matplotlib.")
    args = parser.parse_args()
    status = build_report(args.output, args.expected_jobs)
    if args.plots:
        status["figures"] = build_plots(args.output)
    print(json.dumps(_json_safe(status), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
