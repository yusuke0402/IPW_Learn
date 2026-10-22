"""Prepare this repository's pinned source and frozen standalone pilot runtime.

Python 3.10+ and Git are sufficient for --sources-only. Full setup additionally
requires an installed uv executable, which manages the requested Python 3.12.
All mutable setup state lives under --workdir. No repository checkout is edited.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
PIN_FILE = ROOT / "source_pins.json"
FAMILIES = {"iwl", "aipw", "ipw", "gcomp", "conditional", "commensurate"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic(path, value, mode=0o644):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def setup_lock(workdir):
    with (workdir / ".prepare.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another setup process is using this workdir.") from error
        yield


def environment(workdir):
    result = dict(os.environ)
    result.setdefault("UV_CACHE_DIR", str(workdir / "cache/uv"))
    result.update(MPLCONFIGDIR=str(workdir / "cache/matplotlib"),
                  XDG_CACHE_HOME=str(workdir / "cache"),
                  MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1",
                  GIT_TERMINAL_PROMPT="0")
    result.pop("PYTHONPATH", None)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        result[key] = "1"
    return result


def git(repository, *args):
    return subprocess.check_output(["git", "-C", str(repository), *args], stderr=subprocess.PIPE)


def has_commit(repository, commit):
    try:
        return git(repository, "cat-file", "-t", commit).strip() == b"commit"
    except subprocess.CalledProcessError:
        return False


def repository_for(workdir, family, pin, source_cache):
    if source_cache is not None:
        for candidate in (source_cache, source_cache / pin["repository"].split("/")[-1], source_cache / family):
            if candidate.is_dir() and has_commit(candidate, pin["commit"]):
                return candidate
    # The topic branch includes the scientific pin as an ancestor. This exports
    # Git blobs without touching the user's checkout or consulting other repos.
    own_checkout = ROOT.parents[1]
    if has_commit(own_checkout, pin["commit"]):
        return own_checkout
    repository = workdir / "cache/git" / (family + ".git")
    if not repository.exists():
        repository.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "--bare", "--quiet", str(repository)], check=True)
    if not has_commit(repository, pin["commit"]):
        url = "https://github.com/" + pin["repository"] + ".git"
        print(f"Fetching {pin['repository']} at {pin['commit'][:12]}", flush=True)
        # Fetch by the exact immutable object ID; never resolve a moving branch.
        subprocess.run(["git", "-C", str(repository), "fetch", "--quiet", "--no-tags", "--depth=1", url, pin["commit"]],
                       env=environment(workdir), check=True)
    if not has_commit(repository, pin["commit"]):
        raise RuntimeError(f"Pinned commit could not be obtained: {family}")
    return repository


def wanted(relative):
    path = Path(relative)
    return path.suffix in {".py", ".toml", ".lock", ".yaml", ".yml", ".sh"} or \
        (relative.startswith("configs/") and path.suffix == ".csv") or \
        relative in {".python-version", ".gitignore", "README.md"}


def safe_destination(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe source path: {relative!r}")
    destination = root / path
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Source path leaves its workdir: {relative!r}")
    if destination.is_symlink():
        raise ValueError(f"Refusing source symlink: {relative!r}")
    return destination


def export_source(workdir, family, pin, repository):
    root = workdir / "sources" / family
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for entry in git(repository, "ls-tree", "-rz", pin["commit"]).split(b"\0"):
        if not entry:
            continue
        metadata, relative = entry.split(b"\t", 1)
        mode, kind, blob = metadata.decode().split()
        relative = relative.decode("utf-8")
        if kind != "blob" or mode not in {"100644", "100755"} or not wanted(relative):
            continue
        data = git(repository, "cat-file", "blob", blob)
        entries.append((relative, mode, blob, data))
    files = {relative: dict(sha256=hashlib.sha256(data).hexdigest(), git_blob=blob)
             for relative, mode, blob, data in entries}
    record = dict(path=f"sources/{family}", files=files)
    existing_manifest = root / "SOURCE_MANIFEST.json"
    if existing_manifest.exists() and read(existing_manifest) != record:
        raise ValueError(f"Source manifest differs for {family}; use a new workdir.")
    # Validate every existing file before publishing any missing source files.
    for relative, mode, blob, data in entries:
        path = safe_destination(root, relative)
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Refusing to replace changed source: {family}/{relative}")
    expected_python = {relative for relative in files if relative.endswith(".py")}
    actual_python = {str(path.relative_to(root)) for path in root.rglob("*.py")
                     if ".venv" not in path.relative_to(root).parts and "__pycache__" not in path.parts}
    if actual_python - expected_python:
        raise ValueError(f"Unrecorded Python source files in {family}: {sorted(actual_python - expected_python)}")
    for relative, mode, blob, data in entries:
        path = safe_destination(root, relative)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(0o555 if mode == "100755" else 0o444)
    if not existing_manifest.exists():
        atomic(existing_manifest, record, mode=0o444)
    print(f"{family}: {pin['commit'][:12]}, {len(files)} verified files", flush=True)
    return record


def runtime_info(workdir, relative_python):
    code = ("import json,platform,importlib.metadata as m; "
            "print(json.dumps({'version':platform.python_version(),'implementation':platform.python_implementation(),"
            "'packages':sorted((d.metadata['Name'].lower(),d.version) "
            "for d in m.distributions())}))")
    info = json.loads(subprocess.check_output([str(workdir / relative_python), "-c", code],
                                            env=environment(workdir), text=True))
    info["python"] = relative_python
    return info


def runtime_project(workdir, directory="runtime"):
    """Copy only the repository-supplied lock project; preserve changed files."""
    names = ("pyproject.toml", "uv.lock", ".python-version")
    expected = {}
    for name in names:
        source = ROOT / directory / name
        if not source.is_file():
            raise FileNotFoundError(f"Bundled frozen runtime is missing: {source}")
        expected[name] = dict(sha256=sha(source))
        destination = safe_destination(workdir, directory + "/" + name)
        if destination.exists() and sha(destination) != expected[name]["sha256"]:
            raise ValueError(f"Refusing to replace changed runtime project: {name}")
    for name in names:
        source, destination = ROOT / directory / name, workdir / directory / name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            destination.chmod(0o444)
    return dict(path=directory, files=expected)


def prepare(workdir, python="3.12.14", source_cache=None, sources_only=False):
    workdir = Path(workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    source_cache = None if source_cache is None else Path(source_cache).expanduser().resolve()
    all_pins = read(PIN_FILE)
    design = read(ROOT / "design.json")
    family = design.get("repository_family")
    if family not in FAMILIES:
        raise ValueError("This standalone bundle requires one non-NPP repository family.")
    pins = {family: all_pins[family]}
    report_key = "report" if family in {"iwl", "aipw", "ipw", "gcomp"} else family
    for family, pin in pins.items():
        if len(pin["commit"]) != 40 or any(c not in "0123456789abcdef" for c in pin["commit"]):
            raise ValueError(f"Invalid exact commit for {family}")
        if pin["repository"].count("/") != 1 or any(x in pin["repository"] for x in ("..", ":", "\\")):
            raise ValueError(f"Invalid GitHub repository for {family}")
    if not sources_only and shutil.which("uv") is None:
        raise RuntimeError("uv is required. Install uv using its official platform instructions, then rerun setup.")
    with setup_lock(workdir):
        path = workdir / "prepared.json"
        old = read(path) if path.exists() else None
        if old is not None and (old.get("source_pins") != pins or old.get("python_requested") != python):
            raise ValueError("Existing setup uses different pins or Python; preserve it and use a new workdir.")
        projects = {family: runtime_project(workdir)}
        if report_key == "report":
            projects["report"] = runtime_project(workdir, "reporting")
        if old is not None and old.get("runtime_projects") != projects:
            raise ValueError("Bundled runtime project changed; preserve existing setup and use a new workdir.")
        if old is not None:
            for key, recorded in old.get("runtimes", {}).items():
                executable = workdir / recorded["python"]
                if executable.exists() and runtime_info(workdir, recorded["python"]) != recorded:
                    raise ValueError(f"Existing frozen runtime changed: {key}. Refusing to rewrite its setup record.")
        sources = {}
        for family, pin in pins.items():
            repository = repository_for(workdir, family, pin, source_cache)
            sources[family] = export_source(workdir, family, pin, repository)
        runtimes = old.get("runtimes", {}) if old is not None else {}
        if not sources_only:
            runtimes = {}
            for runtime_key, project in projects.items():
                source = workdir / project["path"]
                log_path = workdir / "logs/setup" / (runtime_key + ".log")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                relative_python = project["path"] + "/.venv/bin/python"
                print(f"Installing {runtime_key} frozen runtime; log: {log_path}", flush=True)
                with log_path.open("a", encoding="utf-8") as log:
                    subprocess.run(["uv", "sync", "--frozen", "--no-install-project", "--python", python], cwd=source,
                                   env=environment(workdir), stdout=log, stderr=subprocess.STDOUT, check=True)
                    subprocess.run(["uv", "pip", "check", "--python", str(workdir / relative_python)],
                                   env=environment(workdir), stdout=log, stderr=subprocess.STDOUT, check=True)
                runtimes[runtime_key] = runtime_info(workdir, relative_python)
        record = dict(schema_version=1, source_pins=pins, sources=sources, runtimes=runtimes,
                      runtime_projects=projects, runtime_for_family={family: family}, python_requested=python,
                      report_runtime=report_key, status="ready" if set(runtimes) == set(projects) else "sources_only")
        if old is not None and old != record and old.get("status") == "ready":
            raise ValueError("Existing prepared environment differs; preserve it and use a new workdir.")
        if old != record:
            atomic(path, record)
        print(f"{record['status']}: {path}", flush=True)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=ROOT / "work")
    parser.add_argument("--python", default="3.12.14")
    parser.add_argument("--source-cache", type=Path, help="Optional local Git checkout parent; exact object IDs are still verified.")
    parser.add_argument("--sources-only", action="store_true", help="Export sources without installing dependencies; runner execution remains unavailable.")
    args = parser.parse_args()
    prepare(args.workdir, args.python, args.source_cache, args.sources_only)


if __name__ == "__main__":
    main()
