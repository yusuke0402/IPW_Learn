"""Run a fixed, portable simulation pilot for this repository.

Source revisions, dependencies and scientific settings are checked before runs
or resume. The coordinator uses only the Python standard library. Each method
runs in its own process and publishes one checksummed checkpoint per seed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

BUNDLE = Path(__file__).resolve().parent
WEIGHTED = [f"{family}_{weight}{suffix}" for family, suffix in
            (("iwl", "_l1none"), ("aipw_sn", ""), ("ipw_sn", ""))
            for weight in ("ulsif", "oracle", "logistic")]
OTHER = ["g_computation", "conditional_pp", "commensurate_pp"]
FAMILY = {method: ("aipw" if method.startswith("aipw_") else
                  "ipw" if method.startswith("ipw_") else "iwl") for method in WEIGHTED}
FAMILY.update(dict(zip(OTHER, ("gcomp", "conditional", "commensurate"))))
FAMILIES = frozenset(FAMILY.values())
BUNDLE_FILES = ("runner.py", "prepare.py", "pilot_weighted.py", "pilot_bayes.py",
                "pilot_report.py", "design.json", "source_pins.json")

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "tolist"):
        return safe(value.tolist())
    if hasattr(value, "item"):
        return safe(value.item())
    return value

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(safe(value), handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def read(path):
    return json.loads(Path(path).read_text())

def job(method, sid, source_n, gamma, seed):
    return dict(method=method, scenario=f"data_scenario_{sid}", scenario_id=int(sid),
                source_n=int(source_n), target_n=50, gamma=float(gamma), seed=int(seed))

def identity(item):
    return (item["method"], item["scenario_id"], item["source_n"], item["gamma"], item["seed"])

def checkpoint(output, item):
    gamma = format(item["gamma"], "g").replace(".", "p")
    return Path(output) / "raw" / item["method"] / f"scenario_{item['scenario_id']:02}" / \
        f"gamma_{gamma}" / f"source_{item['source_n']:03}" / f"trial_{item['seed']:06}.json"

def valid_checkpoint(path, manifest, expected):
    data = read(path)
    checksum = data.get("result_sha256")
    payload = {k: v for k, v in data.items() if k != "result_sha256"}
    if checksum != hashlib.sha256(canonical(payload).encode()).hexdigest():
        raise ValueError(f"Checkpoint content checksum mismatch: {path}")
    if data["fingerprint"] != manifest["fingerprint"] or data["job"] != expected:
        raise ValueError(f"Checkpoint configuration mismatch: {path}")
    rows = data.get("normalized")
    n_rows = 5 if expected["method"] == "conditional_pp" else 1
    if not isinstance(rows, list) or len(rows) != n_rows or not isinstance(data.get("raw"), dict):
        raise ValueError(f"Incomplete checkpoint: {path}")
    digest = data["data_sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"Invalid data hash: {path}")
    seen = set()
    for row in rows:
        if row["data_sha256"] != digest or row["seed"] != expected["seed"] or \
           row["source_n"] != expected["source_n"] or row["target_n"] != 50 or \
           row["gamma"] != expected["gamma"] or row["scenario_id"] != expected["scenario_id"]:
            raise ValueError(f"Normalized row identity mismatch: {path}")
        method = row["method"]
        if method in seen or (method != expected["method"] and not
                              (expected["method"] == "conditional_pp" and method.startswith("conditional_pp_a0_"))):
            raise ValueError(f"Normalized method mismatch: {path}")
        seen.add(method)
        def finite(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        estimate, lower, upper = (row[k] for k in ("estimate", "ci_lower", "ci_upper"))
        point_ok = finite(estimate)
        ci_ok = point_ok and finite(lower) and finite(upper) and lower <= upper
        if row["point_success"] is not point_ok or row["ci_success"] is not ci_ok:
            raise ValueError(f"Result eligibility mismatch: {path}")
        if row["true_value"] != 1. or not row.get("interval_type"):
            raise ValueError(f"Missing scientific result metadata: {path}")
        if expected["method"] in WEIGHTED + ["g_computation"]:
            counts = [row[k] for k in ("boot_requested", "boot_success", "boot_failed")]
            if any(type(v) is not int or v < 0 for v in counts) or counts[0] != 200 or sum(counts[1:]) != 200:
                raise ValueError(f"Bootstrap accounting mismatch: {path}")
            if row["ci_success"] and row["boot_failed"] > 10:
                raise ValueError(f"Ineligible bootstrap CI: {path}")
    if expected["method"] == "conditional_pp":
        methods = {"conditional_pp_a0_" + format(a0, "g").replace(".", "p")
                   for a0 in (.1, .25, .5, .75, 1.)}
        if seen != methods:
            raise ValueError(f"Conditional PP a0 set mismatch: {path}")
    return data

def inventory(output, manifest):
    jobs = jobs_for(manifest)
    planned = {checkpoint(output, item): item for item in jobs}
    actual = set((Path(output) / "raw").rglob("trial_*.json"))
    if actual - planned.keys():
        raise ValueError(f"Unexpected checkpoints found: {next(iter(actual - planned.keys()))}")
    hashes, completed = {}, set()
    for path in sorted(actual):
        item = planned[path]
        data = valid_checkpoint(path, manifest, item)
        key = identity(item)[1:]
        if key in hashes and hashes[key] != data["data_sha256"]:
            raise ValueError(f"Paired input data mismatch: {path}")
        hashes[key] = data["data_sha256"]
        completed.add(identity(item))
    return jobs, completed

@contextmanager
def exclusive(output):
    Path(output).mkdir(parents=True, exist_ok=True)
    with open(Path(output) / ".driver.lock", "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another driver holds this output directory.") from exc
        yield


def relative_path(workdir, relative):
    value = PurePosixPath(relative)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise ValueError(f"Prepared path must be relative to workdir: {relative!r}")
    return Path(workdir).joinpath(*value.parts)


def available_cpus():
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    # Container quota may be smaller than the visible affinity set.
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        amount, period = quota.read_text().split()
        if amount != "max":
            count = min(count, max(1, int(amount) // int(period)))
    else:
        amount = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if amount.exists() and period.exists() and int(amount.read_text()) > 0:
            count = min(count, max(1, int(amount.read_text()) // int(period.read_text())))
    return max(1, count)


def environment(workdir):
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = "1"
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", MPLBACKEND="Agg",
               MPLCONFIGDIR=str(Path(workdir) / "cache/matplotlib"),
               XDG_CACHE_HOME=str(Path(workdir) / "cache"))
    env.pop("PYTHONPATH", None)
    return env


def selected_methods(family):
    if family not in FAMILIES:
        raise ValueError(f"Unknown repository family: {family!r}")
    return [method for method in WEIGHTED + OTHER if FAMILY[method] == family]


def validated_design():
    design = read(BUNDLE / "design.json")
    required = dict(seeds=list(range(200)), trials_per_condition=200,
                    bootstrap_replicates=200, target_n=50, source_ns=[20, 100, 500],
                    gammas=[0., .5], weighted_scenarios=[1, 2, 3, 4], other_scenarios=[4],
                    iwl_l1=0., iwl_lambda_selection="fixed", iwl_target_split_ratio=1.,
                    aipw="self_normalized", ipw="self_normalized", min_ess=3.,
                    max_bootstrap_failure_rate=.05, conditional_a0=[.1, .25, .5, .75, 1.])
    for key, expected in required.items():
        if design.get(key) != expected:
            raise ValueError(f"Unsupported pilot design {key!r}; expected {expected!r}.")
    selected_methods(design.get("repository_family"))
    return design


def production_jobs(family):
    for method in selected_methods(family):
        for sid, source_n, gamma, seed in itertools.product(
                (1, 2, 3, 4) if method in WEIGHTED else (4,),
                (20, 100, 500), (0., .5), range(200)):
            yield job(method, sid, source_n, gamma, seed)


def preflight_jobs(family):
    for method in selected_methods(family):
        yield job(method, 4, 20, 0., 0)


def jobs_for(manifest):
    return list(production_jobs(manifest["repository_family"])) if manifest["mode"] == "production" \
        else manifest["verification_jobs"]


def make_manifest(workdir, mode="production"):
    workdir = Path(workdir).resolve()
    design = validated_design()
    family = design["repository_family"]
    prepared = read(workdir / "prepared.json")
    if prepared.get("status") != "ready" or set(prepared.get("sources", {})) != {family}:
        raise ValueError("Run prepare.py for this repository before preparing an experiment.")
    expected_pin = {family: read(BUNDLE / "source_pins.json")[family]}
    if prepared.get("source_pins") != expected_pin:
        raise ValueError("Prepared source revision differs from this repository's pinned revision.")
    if set(prepared.get("runtime_for_family", {})) != {family}:
        raise ValueError("Prepared runtime does not match this repository.")
    content = dict(schema_version=1, mode=mode, repository_family=family,
                   requested_design=design, prepared=prepared,
                   scripts={name: sha(BUNDLE / name) for name in BUNDLE_FILES})
    for project in prepared["runtime_projects"].values():
        for name in project["files"]:
            relative = str(PurePosixPath(project["path"]) / name)
            content["scripts"][relative] = sha(relative_path(BUNDLE, relative))
    if mode == "verification":
        content["verification_jobs"] = list(preflight_jobs(family))
        content["research_interpretation"] = "execution verification only; not research results"
    elif mode != "production":
        raise ValueError("Unknown execution mode.")
    jobs = jobs_for(content)
    content["expected_jobs"] = len(jobs)
    content["expected_normalized_rows"] = sum(5 if item["method"] == "conditional_pp" else 1 for item in jobs)
    content["fingerprint"] = hashlib.sha256(canonical(content).encode()).hexdigest()
    return content


def validate_manifest(manifest, workdir, family=None):
    workdir = Path(workdir).resolve()
    content = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if hashlib.sha256(canonical(content).encode()).hexdigest() != manifest["fingerprint"]:
        raise ValueError("Manifest fingerprint mismatch: refuse changed settings.")
    for name, digest in manifest["scripts"].items():
        if sha(BUNDLE / name) != digest:
            raise ValueError(f"Execution code or configuration changed: {name}; old results will not be reused.")
    prepared = manifest["prepared"]
    if read(workdir / "prepared.json") != prepared:
        raise ValueError("Prepared source or runtime metadata changed; old results will not be reused.")
    if family is not None and family != manifest["repository_family"]:
        raise ValueError("Worker family differs from the saved experiment.")
    for key, record in prepared["sources"].items():
        source = relative_path(workdir, record["path"])
        for relative, item in record["files"].items():
            if sha(relative_path(source, relative)) != item["sha256"]:
                raise ValueError(f"Pinned source changed: {key}/{relative}")
        actual = {str(path.relative_to(source).as_posix()) for path in source.rglob("*.py")
                  if not {".venv", ".git", "__pycache__"}.intersection(path.relative_to(source).parts)}
        expected = {relative for relative in record["files"] if relative.endswith(".py")}
        if actual != expected:
            raise ValueError(f"Unrecorded Python source files: {key}")
    for project in prepared["runtime_projects"].values():
        runtime_root = relative_path(workdir, project["path"])
        for relative, item in project["files"].items():
            if sha(relative_path(runtime_root, relative)) != item["sha256"]:
                raise ValueError(f"Frozen runtime project changed: {relative}")
    from prepare import runtime_info
    for key, recorded in prepared["runtimes"].items():
        if runtime_info(workdir, recorded["python"]) != recorded:
            raise ValueError(f"Frozen runtime packages or Python changed: {key}")


def initialize(output, manifest):
    output = Path(output)
    path = output / "manifest.json"
    if path.exists():
        if read(path) != manifest:
            raise ValueError(f"Existing manifest differs: {path}. Preserve results and use a new output directory.")
    else:
        if list((output / "raw").rglob("trial_*.json")):
            raise ValueError("Existing raw results without a manifest.")
        atomic(path, manifest)
    return manifest


def write_plan(output, manifest):
    conditions = {}
    for item in jobs_for(manifest):
        key = identity(item)[:-1]
        conditions.setdefault(key, []).append(item["seed"])
    rows = []
    for (method, scenario, source_n, gamma), seeds in conditions.items():
        rows.append(dict(method=method, scenario=scenario, gamma=gamma, target_n=50,
                         source_n=source_n, seed_start=min(seeds), seed_end=max(seeds),
                         trials=len(seeds), bootstrap_replicates=200 if method in WEIGHTED + ["g_computation"] else 0,
                         normalized_outputs_per_trial=5 if method == "conditional_pp" else 1))
    path = Path(output) / "planned_conditions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".planned_conditions", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def worker(workdir, output, batch_file):
    manifest = read(Path(output) / "manifest.json")
    batch = read(batch_file)
    family = manifest["repository_family"]
    if not batch or any(item["method"] != batch[0]["method"] for item in batch):
        raise ValueError("A worker batch must use exactly one method.")
    if FAMILY.get(batch[0]["method"]) != family:
        raise ValueError("Worker method differs from this repository.")
    validate_manifest(manifest, workdir, family)
    prepared = manifest["prepared"]
    runtime_key = prepared["runtime_for_family"][family]
    python = relative_path(workdir, prepared["runtimes"][runtime_key]["python"])
    if Path(sys.executable).absolute() != python.absolute():
        raise ValueError("Worker launched in the wrong runtime.")
    if family in ("conditional", "commensurate"):
        from pilot_bayes import run_trial
    else:
        from pilot_weighted import run_trial
    context = dict(mode=manifest["mode"], sources={key: str(relative_path(workdir, value["path"]))
                   for key, value in prepared["sources"].items()}, bootstrap_replicates=200,
                   target_n=50, min_ess=3., max_bootstrap_failure_rate=.05)
    for item in batch:
        if (Path(output) / "STOP").exists():
            break
        path = checkpoint(output, item)
        if path.exists():
            valid_checkpoint(path, manifest, item)
            continue
        started = time.monotonic()
        try:
            result = run_trial(item["method"], item["scenario"], item["source_n"], item["gamma"], item["seed"], context)
            record = safe(dict(job=item, fingerprint=manifest["fingerprint"], **result,
                               seconds=time.monotonic() - started, completed_at_utc=utc()))
            record["result_sha256"] = hashlib.sha256(canonical(record).encode()).hexdigest()
            pending = path.with_name(".candidate_" + path.name)
            atomic(pending, record)
            try:
                valid_checkpoint(pending, manifest, item)
                os.replace(pending, path)
            finally:
                pending.unlink(missing_ok=True)
            print(canonical(dict(completed=identity(item), seconds=record["seconds"])), flush=True)
        except Exception:
            atomic(Path(output) / "errors" / (item["method"] + "_" + uuid.uuid4().hex + ".json"),
                   dict(job=item, at_utc=utc(), traceback=traceback.format_exc()))
            raise


def report(workdir, output, plots=False):
    output = Path(output)
    manifest = read(output / "manifest.json")
    validate_manifest(manifest, workdir)
    inventory(output, manifest)
    with (output / ".report.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another report writer holds this output directory.") from exc
        prepared = manifest["prepared"]
        key = prepared["report_runtime"]
        python = relative_path(workdir, prepared["runtimes"][key]["python"])
        command = [str(python), str(BUNDLE / "pilot_report.py"), "--output", str(output)]
        if plots:
            command.append("--plots")
        subprocess.run(command, env=environment(workdir), check=True)


def execute(workdir, output, workers=None, batch_size=10, clear_stop=False, plots=True):
    workdir, output = Path(workdir).resolve(), Path(output).resolve()
    workers = available_cpus() if workers is None else workers
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch_size must be positive integers.")
    with exclusive(output):
        with (output / ".workers.lock").open("a+") as lease_check:
            try:
                fcntl.flock(lease_check, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Workers from an earlier driver are active; wait for their checkpoints.") from exc
        manifest = read(output / "manifest.json")
        validate_manifest(manifest, workdir)
        jobs, completed = inventory(output, manifest)
        paired_hashes = {identity(item)[1:]: read(checkpoint(output, item))["data_sha256"]
                         for item in jobs if identity(item) in completed}
        if clear_stop:
            (output / "STOP").unlink(missing_ok=True)
        if (output / "STOP").exists():
            raise ValueError("STOP exists; resume explicitly with run --clear-stop.")
        grouped = {}
        for item in jobs:
            if identity(item) not in completed:
                grouped.setdefault(identity(item)[:-1], []).append(item)
        batches = [group[index:index + batch_size] for group in grouped.values()
                   for index in range(0, len(group), batch_size)]
        session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
        batch_root = output / "jobs" / session
        log_root = output / "logs" / session
        batch_root.mkdir(parents=True)
        log_root.mkdir(parents=True)
        started = time.monotonic()
        state = dict(session=session, started_at_utc=utc(), mode=manifest["mode"],
                     repository_family=manifest["repository_family"], jobs_total=len(jobs),
                     jobs_already_complete=len(completed), jobs_complete=len(completed),
                     jobs_pending=len(jobs) - len(completed), workers=workers,
                     batch_size=batch_size, available_cpu_count=available_cpus(), status="running")
        errors, now_complete = [], set(completed)

        def publish():
            state.update(updated_at_utc=utc(), elapsed_seconds=time.monotonic() - started,
                         jobs_complete=len(now_complete), jobs_pending=len(jobs) - len(now_complete),
                         jobs_new=len(now_complete) - len(completed), errors=errors)
            atomic(output / "progress.json", state)

        publish()
        old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

        def request_stop(signum, frame):
            atomic(output / "STOP", dict(reason=signal.Signals(signum).name, at_utc=utc()))
            state["status"] = "stopping"
            print("Stop requested; saving current trials before exit.", flush=True)

        for sig in old_handlers:
            signal.signal(sig, request_stop)

        def launch(index, batch):
            descriptor = batch_root / f"batch_{index:06}.json"
            atomic(descriptor, batch)
            prepared = manifest["prepared"]
            key = prepared["runtime_for_family"][manifest["repository_family"]]
            python = relative_path(workdir, prepared["runtimes"][key]["python"])
            logfile = log_root / f"batch_{index:06}_{batch[0]['method']}.log"
            with logfile.open("w") as handle, (output / ".workers.lock").open("a+") as lease:
                fcntl.flock(lease, fcntl.LOCK_SH)
                process = subprocess.Popen([str(python), str(BUNDLE / "runner.py"), "_worker",
                                            "--workdir", str(workdir), "--output", str(output),
                                            "--batch-file", str(descriptor)], env=environment(workdir),
                                           stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                           start_new_session=True, pass_fds=(lease.fileno(),))
                return process.wait(), str(logfile.relative_to(output))

        queue, active, index = iter(batches), {}, 0
        heartbeat = time.monotonic()
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            def submit():
                nonlocal index
                if (output / "STOP").exists():
                    return
                batch = next(queue, None)
                if batch is None:
                    return
                future = pool.submit(launch, index, batch)
                active[future] = batch
                index += 1

            for _ in range(workers):
                submit()
            while active:
                done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    batch = active.pop(future)
                    try:
                        code, logfile = future.result()
                        for item in batch:
                            path = checkpoint(output, item)
                            if not path.exists():
                                if code or (output / "STOP").exists():
                                    continue
                                raise ValueError(f"Worker omitted a checkpoint: {path}")
                            data = valid_checkpoint(path, manifest, item)
                            key = identity(item)[1:]
                            if key in paired_hashes and paired_hashes[key] != data["data_sha256"]:
                                raise ValueError(f"New paired input data mismatch: {path}")
                            paired_hashes[key] = data["data_sha256"]
                            now_complete.add(identity(item))
                        if code:
                            errors.append(dict(exit_code=code, log=logfile))
                            atomic(output / "STOP", dict(reason="unexpected_worker_error", errors=errors, at_utc=utc()))
                    except Exception:
                        errors.append(dict(traceback=traceback.format_exc()))
                        atomic(output / "STOP", dict(reason="driver_error", errors=errors, at_utc=utc()))
                    publish()
                    submit()
                if time.monotonic() - heartbeat >= 15:
                    # Long batches still save each seed; expose those counts.
                    state["jobs_saved"] = sum(1 for _ in (output / "raw").rglob("trial_*.json"))
                    if (output / "STOP").exists():
                        state["status"] = "stopping"
                    publish()
                    heartbeat = time.monotonic()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        _, now_complete = inventory(output, manifest)
        state.update(finished_at_utc=utc(), status="failed" if errors else
                     "complete" if len(now_complete) == len(jobs) else "stopped")
        publish()
        print(json.dumps(state, ensure_ascii=False, indent=2))
        if errors:
            raise RuntimeError(f"Worker failed; inspect {errors[0].get('log', 'errors/')} below {output}.")
        if state["status"] == "complete":
            try:
                report(workdir, output, plots=plots)
            except Exception:
                state.update(status="report_failed", report_error=traceback.format_exc())
                publish()
                raise
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "plan", "preflight", "run", "status", "stop", "report", "_worker"))
    parser.add_argument("--workdir", type=Path, default=BUNDLE / "work")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--clear-stop", action="store_true")
    parser.add_argument("--plots", action="store_true", help="Create plots in addition to CSV reports (requires matplotlib).")
    parser.add_argument("--batch-file", type=Path)
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1 or args.batch_size < 1:
        parser.error("workers and batch-size must be positive integers")
    workdir = args.workdir.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output is not None else workdir / "results"
    if args.command == "stop":
        atomic(output / "STOP", dict(reason="user_stop", at_utc=utc()))
        print("Stop requested; workers finish and save their current trial.")
        return
    if args.command == "_worker":
        worker(workdir, output, args.batch_file)
        return
    if args.command == "prepare":
        manifest = make_manifest(workdir)
        validate_manifest(manifest, workdir)
        with exclusive(output):
            initialize(output, manifest)
            write_plan(output, manifest)
        print(canonical(dict(manifest=str(output / "manifest.json"), jobs=manifest["expected_jobs"],
                             normalized_rows=manifest["expected_normalized_rows"], fingerprint=manifest["fingerprint"])))
    elif args.command == "preflight":
        output = args.output.expanduser().resolve() if args.output is not None else workdir / "verification/preflight"
        if not output.is_relative_to(workdir / "verification"):
            raise ValueError("Preflight output must be below workdir/verification, separate from research results.")
        manifest = make_manifest(workdir, "verification")
        validate_manifest(manifest, workdir)
        with exclusive(output):
            initialize(output, manifest)
            write_plan(output, manifest)
        execute(workdir, output, args.workers, args.batch_size, args.clear_stop, plots=True)
    elif args.command == "run":
        execute(workdir, output, args.workers, args.batch_size, args.clear_stop, plots=True)
    elif args.command == "report":
        report(workdir, output, args.plots)
    else:
        manifest = read(output / "manifest.json")
        validate_manifest(manifest, workdir)
        jobs, completed = inventory(output, manifest)
        print(json.dumps(dict(mode=manifest["mode"], repository_family=manifest["repository_family"],
                              jobs_total=len(jobs), jobs_complete=len(completed), jobs_remaining=len(jobs) - len(completed),
                              normalized_rows_expected=manifest["expected_normalized_rows"], available_cpu_count=available_cpus(),
                              fingerprint=manifest["fingerprint"], stop_requested=(output / "STOP").exists()), indent=2))


if __name__ == "__main__":
    main()
