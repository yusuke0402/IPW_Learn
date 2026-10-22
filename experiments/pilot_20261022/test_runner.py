"""Exercise portability and resume boundaries using disposable files, without fits.

Run ``python -m unittest discover -s experiments/pilot_20261022 -p test_runner.py``.
The tests never alter scientific source exports, installed runtimes or results.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import prepare
import runner


class PilotBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pilot_boundary_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "bundle"
        self.workdir = self.root / "work"
        self.output = self.workdir / "results"
        self.bundle.mkdir()
        self.output.mkdir(parents=True)
        self.original_bundle = runner.BUNDLE
        self.bundle_patch = mock.patch.object(runner, "BUNDLE", self.bundle)
        self.bundle_patch.start()
        self.addCleanup(self.bundle_patch.stop)
        self.source = self.workdir / "sources/iwl/model.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("# exported scientific source fixture\n")
        self.project = self.workdir / "runtime/pyproject.toml"
        self.project.parent.mkdir(parents=True)
        self.project.write_text("# frozen runtime fixture\n")
        self.launcher = self.bundle / "fixture.py"
        self.launcher.write_text("# execution fixture\n")
        self.runtime = dict(python="runtime/.venv/bin/python", version="3.12.14",
                            implementation="CPython", packages=[["numpy", "2.5.3"]])
        self.prepared = dict(status="ready", source_pins={"iwl": dict(repository="owner/repo", commit="a" * 40)},
                             sources={"iwl": dict(path="sources/iwl", files={"model.py": dict(sha256=runner.sha(self.source))})},
                             runtime_projects={"iwl": dict(path="runtime", files={"pyproject.toml": dict(sha256=runner.sha(self.project))})},
                             runtimes={"iwl": self.runtime}, runtime_for_family={"iwl": "iwl"}, report_runtime="iwl")
        runner.atomic(self.workdir / "prepared.json", self.prepared)
        self.items = [runner.job("iwl_oracle_l1none", 4, 20, 0., 0),
                      runner.job("iwl_logistic_l1none", 4, 20, 0., 0)]
        self.manifest = dict(schema_version=1, mode="verification", repository_family="iwl",
                             prepared=self.prepared, scripts={"fixture.py": runner.sha(self.launcher)},
                             verification_jobs=self.items, expected_jobs=2, expected_normalized_rows=2)
        self.sign_manifest(self.manifest)
        self.runtime_patch = mock.patch.object(prepare, "runtime_info", return_value=self.runtime)
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)

    @staticmethod
    def sign_manifest(manifest):
        content = {key: value for key, value in manifest.items() if key != "fingerprint"}
        manifest["fingerprint"] = hashlib.sha256(runner.canonical(content).encode()).hexdigest()

    @staticmethod
    def sign_record(record):
        record.pop("result_sha256", None)
        record["result_sha256"] = hashlib.sha256(runner.canonical(record).encode()).hexdigest()
        return record

    def record(self, item=None, digest="a" * 64):
        item = copy.deepcopy(self.items[0] if item is None else item)
        row = dict(item, data_sha256=digest, estimate=1., ci_lower=.8, ci_upper=1.2,
                   true_value=1., interval_type="bootstrap_percentile_95", point_success=True,
                   ci_success=True, boot_requested=200, boot_success=200, boot_failed=0)
        return self.sign_record(dict(job=item, fingerprint=self.manifest["fingerprint"],
                                     data_sha256=digest, raw=dict(trial=item["seed"]), normalized=[row]))

    def publish(self, record, resign=False):
        if resign:
            self.sign_record(record)
        path = runner.checkpoint(self.output, record["job"])
        runner.atomic(path, record)
        return path

    def test_repository_plans_cover_exact_requested_matrix(self):
        expected = dict(iwl=(14400, 14400), aipw=(14400, 14400), ipw=(14400, 14400),
                        gcomp=(1200, 1200), conditional=(1200, 6000), commensurate=(1200, 1200))
        for family, (raw_count, row_count) in expected.items():
            jobs = list(runner.production_jobs(family))
            with self.subTest(family=family):
                self.assertEqual(len(jobs), raw_count)
                self.assertEqual(len({runner.identity(item) for item in jobs}), raw_count)
                self.assertEqual(sum(5 if item["method"] == "conditional_pp" else 1 for item in jobs), row_count)
                self.assertEqual({item["seed"] for item in jobs}, set(range(200)))
                self.assertEqual({item["target_n"] for item in jobs}, {50})
                self.assertEqual({item["source_n"] for item in jobs}, {20, 100, 500})
                self.assertEqual({item["gamma"] for item in jobs}, {0., .5})
                self.assertEqual({item["scenario_id"] for item in jobs}, {1, 2, 3, 4} if family in {"iwl", "aipw", "ipw"} else {4})

    def test_preflight_is_bounded_and_repository_specific(self):
        for family in runner.FAMILIES:
            jobs = list(runner.preflight_jobs(family))
            self.assertEqual(len(jobs), 3 if family in {"iwl", "aipw", "ipw"} else 1)
            for item in jobs:
                self.assertEqual(runner.FAMILY[item["method"]], family)
                self.assertEqual((item["scenario_id"], item["source_n"], item["gamma"], item["seed"]), (4, 20, 0., 0))

    def test_cpu_default_obeys_quota_instead_of_visible_count(self):
        class Quota:
            def __init__(self, text):
                self.text = text

            def exists(self):
                return True

            def read_text(self):
                return self.text

        with mock.patch.object(runner.os, "sched_getaffinity", return_value=set(range(8)), create=True):
            with mock.patch.object(runner, "Path", return_value=Quota("400000 100000")):
                self.assertEqual(runner.available_cpus(), 4)
            with mock.patch.object(runner, "Path", return_value=Quota("max 100000")):
                self.assertEqual(runner.available_cpus(), 8)
            with mock.patch.object(runner, "Path", return_value=Quota("50000 100000")):
                self.assertEqual(runner.available_cpus(), 1)

    def test_manifest_source_and_runtime_survive_workdir_relocation(self):
        runner.validate_manifest(self.manifest, self.workdir)
        destination = self.root / "relocated"
        shutil.copytree(self.workdir, destination)
        runner.validate_manifest(self.manifest, destination)
        self.assertNotIn(str(self.workdir), runner.canonical(self.manifest))

    def test_changed_settings_and_code_are_refused(self):
        changed = copy.deepcopy(self.manifest)
        changed["expected_jobs"] = 999
        with self.assertRaisesRegex(ValueError, "Manifest fingerprint mismatch"):
            runner.validate_manifest(changed, self.workdir)
        self.launcher.write_text("# changed execution code\n")
        with self.assertRaisesRegex(ValueError, "Execution code or configuration changed"):
            runner.validate_manifest(self.manifest, self.workdir)

    def test_changed_source_or_runtime_project_is_refused(self):
        original = self.source.read_bytes()
        self.source.write_text("# changed scientific source\n")
        with self.assertRaisesRegex(ValueError, "Pinned source changed"):
            runner.validate_manifest(self.manifest, self.workdir)
        self.source.write_bytes(original)
        self.project.write_text("# changed dependency declarations\n")
        with self.assertRaisesRegex(ValueError, "Frozen runtime project changed"):
            runner.validate_manifest(self.manifest, self.workdir)

    def test_runtime_drift_or_prepared_pin_change_is_refused(self):
        with mock.patch.object(prepare, "runtime_info", return_value=dict(self.runtime, version="3.12.15")):
            with self.assertRaisesRegex(ValueError, "Frozen runtime packages or Python changed"):
                runner.validate_manifest(self.manifest, self.workdir)
        changed = copy.deepcopy(self.prepared)
        changed["source_pins"]["iwl"]["commit"] = "b" * 40
        runner.atomic(self.workdir / "prepared.json", changed)
        with self.assertRaisesRegex(ValueError, "Prepared source or runtime metadata changed"):
            runner.validate_manifest(self.manifest, self.workdir)

    def test_stale_manifest_or_orphan_results_are_not_overwritten(self):
        runner.initialize(self.output, self.manifest)
        original = (self.output / "manifest.json").read_bytes()
        changed = copy.deepcopy(self.manifest)
        changed["fingerprint"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "Existing manifest differs"):
            runner.initialize(self.output, changed)
        self.assertEqual((self.output / "manifest.json").read_bytes(), original)
        (self.output / "manifest.json").unlink()
        self.publish(self.record())
        with self.assertRaisesRegex(ValueError, "without a manifest"):
            runner.initialize(self.output, self.manifest)

    def test_content_mutation_and_wrong_job_cannot_be_reused(self):
        record = self.record()
        path = self.publish(record)
        runner.valid_checkpoint(path, self.manifest, self.items[0])
        record["raw"]["trial"] = 99
        self.publish(record)
        with self.assertRaisesRegex(ValueError, "content checksum mismatch"):
            runner.valid_checkpoint(path, self.manifest, self.items[0])
        record = self.record()
        record["job"]["seed"] = 99
        runner.atomic(path, self.sign_record(record))
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            runner.valid_checkpoint(path, self.manifest, self.items[0])

    def test_bootstrap_five_percent_boundary_and_failure_eligibility(self):
        record = self.record()
        row = record["normalized"][0]
        row.update(boot_success=190, boot_failed=10)
        path = self.publish(record, resign=True)
        runner.valid_checkpoint(path, self.manifest, self.items[0])
        row.update(boot_success=189, boot_failed=11)
        self.publish(record, resign=True)
        with self.assertRaisesRegex(ValueError, "Ineligible bootstrap CI"):
            runner.valid_checkpoint(path, self.manifest, self.items[0])
        row.update(ci_lower=None, ci_upper=None, ci_success=False)
        self.publish(record, resign=True)
        runner.valid_checkpoint(path, self.manifest, self.items[0])
        row.update(estimate=None)
        self.publish(record, resign=True)
        with self.assertRaisesRegex(ValueError, "eligibility mismatch"):
            runner.valid_checkpoint(path, self.manifest, self.items[0])

    def test_conditional_outputs_require_exact_five_a0_values(self):
        item = runner.job("conditional_pp", 4, 20, 0., 0)
        record = self.record(item)
        template = record["normalized"][0]
        record["normalized"] = [dict(template, method="conditional_pp_a0_" + label)
                                for label in ("0p1", "0p25", "0p5", "0p75", "1")]
        path = self.publish(record, resign=True)
        runner.valid_checkpoint(path, self.manifest, item)
        record["normalized"][-1]["method"] = "conditional_pp_a0_0p9"
        self.publish(record, resign=True)
        with self.assertRaisesRegex(ValueError, "a0 set mismatch"):
            runner.valid_checkpoint(path, self.manifest, item)

    def test_pairing_mismatch_blocks_resume(self):
        for item in self.items:
            self.publish(self.record(item))
        _, complete = runner.inventory(self.output, self.manifest)
        self.assertEqual(len(complete), 2)
        self.publish(self.record(self.items[1], digest="b" * 64))
        with self.assertRaisesRegex(ValueError, "Paired input data mismatch"):
            runner.inventory(self.output, self.manifest)

    def test_double_driver_is_refused_and_lock_releases(self):
        with runner.exclusive(self.output):
            with self.assertRaisesRegex(RuntimeError, "Another driver holds"):
                with runner.exclusive(self.output):
                    self.fail("Second driver acquired the output lock")
        with runner.exclusive(self.output):
            pass

    def test_child_keeps_orphan_worker_lease_after_parent_closes_fd(self):
        lease_path = self.output / ".workers.lock"
        lease = lease_path.open("a+")
        fcntl.flock(lease, fcntl.LOCK_SH)
        child = subprocess.Popen([sys.executable, "-c", "import sys;print('ready',flush=True);sys.stdin.read()"],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                 pass_fds=(lease.fileno(),))
        lease.close()
        try:
            self.assertEqual(child.stdout.readline().strip(), "ready")
            with self.assertRaisesRegex(RuntimeError, "Workers from an earlier driver are active"):
                runner.execute(self.workdir, self.output, workers=1)
        finally:
            child.stdin.close()
            child.wait(timeout=10)
            child.stdout.close()
        with lease_path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


if __name__ == "__main__":
    unittest.main()
