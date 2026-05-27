# Copyright (C) 2015-2026 Regents of the University of California
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Slurm mount-recovery integration tests (sortTest-style).

Runs small Toil workflows on a **real** Slurm cluster to verify storage/OSError
recovery and optional in-place ``Alternate=`` partition switching.

Prerequisites
-------------

Set on the cluster head node (paths must be visible from compute nodes where
applicable):

- ``TOIL_TEST_INTEGRATIVE=True``
- ``TOIL_TEST_SLURM_SHARED_DIR`` — shared directory for ``--batchLogsDir`` and job store
- ``TOIL_TEST_SLURM_COORD_DIR`` — coordination directory (node-local or fast disk per site)
- ``TOIL_SLURM_PARTITION`` — target partition (or pass ``--slurmPartition``)

Optional:

- ``TOIL_SLURM_PARTITION_FAILOVER=partA,partB`` — for failover rotation test
- ``TOIL_SLURM_TEST_PARTITION_SWITCH=1`` — observational partition-switch test
- ``TOIL_SLURM_TEST_DEBUG=1`` — DEBUG logging on the leader

Example::

    export TOIL_TEST_INTEGRATIVE=True
    export TOIL_TEST_SLURM_SHARED_DIR=/shared/toil-tests
    export TOIL_TEST_SLURM_COORD_DIR=/local/toil-coord
    export TOIL_SLURM_PARTITION=your_partition

    ./venv/bin/python -m pytest src/toil/test/sort/slurmSortTest.py -v -s

Partition-switch observation (after the cluster can requeue jobs with high
``Restarts``)::

    export TOIL_SLURM_TEST_PARTITION_SWITCH=1
    export TOIL_SLURM_JOB_RESTART_THRESHOLD=2
    export TOIL_SLURM_PARTITION_SWITCH_COOLDOWN=0
    export TOIL_SLURM_PARTITION_SWITCH_POLL_INTERVAL=0.25
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import unittest
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

from toil import resolveEntryPoint
from toil.common import Toil
from toil.exceptions import FailedJobsException
from toil.job import Job
from toil.jobStores.abstractJobStore import NoSuchJobStoreException
from toil.lib.bioio import root_logger
from toil.test import ToilTest, integrative, needs_slurm, slow
from toil.test.sort import slurm_recovery_workflow
from toil.test.sort.sort import makeFileToSort
from toil.test.sort.sortTest import (
    defaultLineLen,
    defaultLines,
    defaultN,
    runMain,
)

logger = logging.getLogger(__name__)

_SHARED_DIR_ENV = "TOIL_TEST_SLURM_SHARED_DIR"
_COORD_DIR_ENV = "TOIL_TEST_SLURM_COORD_DIR"
_PARTITION_SWITCH_ENV = "TOIL_SLURM_TEST_PARTITION_SWITCH"

_LOG_STORAGE_IO = "Storage I/O failure"
_LOG_EXCLUDE_NODES = "Excluding Slurm nodes"
_LOG_FAILOVER = "Slurm partition failover: subsequent worker jobs will use partition"
_LOG_PARTITION_SWITCH = "switched to alternate partition"


def _require_cluster_paths() -> tuple[str, str]:
    shared = os.environ.get(_SHARED_DIR_ENV, "").strip()
    coord = os.environ.get(_COORD_DIR_ENV, "").strip()
    if not shared:
        raise unittest.SkipTest(
            f"Set {_SHARED_DIR_ENV} to a shared path for batch logs and job store"
        )
    if not coord:
        raise unittest.SkipTest(
            f"Set {_COORD_DIR_ENV} to a coordination directory path on workers"
        )
    os.makedirs(shared, exist_ok=True)
    os.makedirs(coord, exist_ok=True)
    return shared, coord


def _slurm_partition() -> str | None:
    return os.environ.get("TOIL_SLURM_PARTITION", "").strip() or None


def _partition_has_alternate(partition: str) -> bool:
    try:
        result = subprocess.run(
            ["scontrol", "show", "partition", partition],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise unittest.SkipTest(f"Could not query partition {partition}: {e}") from e
    return bool(re.search(r"\bAlternate=\S+", result.stdout))


@integrative
@needs_slurm
class SlurmSortTest(ToilTest):
    """End-to-end Slurm workflows for mount recovery and partition handling."""

    def setUp(self) -> None:
        super().setUp()
        self.tempDir = self._createTempDir(purpose="slurmSortTest")
        self.shared_dir, self.coord_dir = _require_cluster_paths()
        self.leader_log_path = os.path.join(self.tempDir, "leader.log")
        partition = _slurm_partition()
        if partition:
            os.environ.setdefault("TOIL_SLURM_PARTITION", partition)

    def tearDown(self) -> None:
        if os.path.exists(self.tempDir):
            shutil.rmtree(self.tempDir)
        ToilTest.tearDown(self)

    @contextmanager
    def _leader_log_capture(self):
        handler = logging.FileHandler(self.leader_log_path)
        handler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            yield self.leader_log_path
        finally:
            root.removeHandler(handler)
            handler.close()

    def _read_leader_log(self) -> str:
        if os.path.exists(self.leader_log_path):
            with open(self.leader_log_path, encoding="utf-8") as handle:
                return handle.read()
        return ""

    def _job_store_locator(self) -> str:
        run_dir = os.path.join(self.shared_dir, f"slurm-sort-{uuid4()}")
        os.makedirs(run_dir, exist_ok=True)
        return f"file:{run_dir}"

    def _batch_logs_dir(self) -> str:
        path = os.path.join(self.shared_dir, f"batch-logs-{uuid4()}")
        os.makedirs(path, exist_ok=True)
        return path

    def _base_slurm_options(self, job_store_locator: str) -> Any:
        options = Job.Runner.getDefaultOptions(job_store_locator)
        options.logLevel = logging.getLevelName(root_logger.getEffectiveLevel())
        if os.environ.get("TOIL_SLURM_TEST_DEBUG", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            options.logLevel = "DEBUG"
        options.batchSystem = "slurm"
        options.clean = "never"
        options.retry_backoff_seconds = 0
        options.realTimeLogging = True
        options.batch_logs_dir = self._batch_logs_dir()
        options.coordination_dir = self.coord_dir
        return options

    def _run_slurm_workflow(
        self,
        main_fn: Callable[..., None],
        *,
        retry_count: int = 2,
        **option_overrides: Any,
    ) -> str:
        """
        Run a workflow entry point on Slurm and return captured leader log text.

        Cleans the job store in ``finally``.
        """
        job_store_locator = self._job_store_locator()
        options = self._base_slurm_options(job_store_locator)
        options.retryCount = retry_count
        for key, value in option_overrides.items():
            setattr(options, key, value)

        log_text = ""
        try:
            with self._leader_log_capture():
                main_fn(options)
            log_text = self._read_leader_log()
        except FailedJobsException as e:
            log_text = self._read_leader_log()
            raise AssertionError(
                f"Workflow failed with {e.numberOfFailedJobs} failed job(s). "
                f"Leader log tail:\n{log_text[-8000:]}"
            ) from e
        finally:
            subprocess.check_call(
                [resolveEntryPoint("toil"), "clean", job_store_locator]
            )
            self.assertRaises(
                NoSuchJobStoreException, Toil.resumeJobStore, job_store_locator
            )
        return log_text

    def _assert_log_contains_any(self, log_text: str, *needles: str) -> None:
        missing = [n for n in needles if n not in log_text]
        self.assertFalse(
            missing,
            f"Leader log missing expected substring(s) {missing}. "
            f"Log tail:\n{log_text[-8000:]}",
        )

    @slow
    def test_slurm_sort_smoke(self) -> None:
        """Merge-sort smoke test on Slurm (same pattern as SortTest.testFileSingle)."""
        job_store_locator = self._job_store_locator()
        output_file = os.path.join(self.tempDir, "sortedFile.txt")
        input_file = os.path.join(self.tempDir, "fileToSort.txt")
        options = self._base_slurm_options(job_store_locator)
        options.retryCount = 2
        options.N = defaultN
        options.outputFile = output_file
        options.fileToSort = input_file
        options.overwriteOutput = True
        options.numLines = defaultLines
        options.lineLength = defaultLineLen

        makeFileToSort(input_file, lines=defaultLines, lineLen=defaultLineLen)
        with open(input_file, encoding="utf-8") as handle:
            expected = handle.readlines()
        expected.sort()

        with self._leader_log_capture():
            with runMain(options):
                pass
        with open(output_file, encoding="utf-8") as handle:
            self.assertEqual(expected, handle.readlines())

        subprocess.check_call([resolveEntryPoint("toil"), "clean", job_store_locator])

    def test_slurm_storage_failure_worker_oserror(self) -> None:
        """Worker OSError on coordination path triggers storage recovery and retry."""
        batch_logs = self._batch_logs_dir()

        def run(options: Any) -> None:
            options.mode = "worker_oserror"
            options.coordinationPath = self.coord_dir
            options.batchLogsPath = batch_logs
            slurm_recovery_workflow.main(options)

        log_text = self._run_slurm_workflow(run, retry_count=2)
        self._assert_log_contains_any(log_text, _LOG_STORAGE_IO, _LOG_EXCLUDE_NODES)

    def test_slurm_storage_failure_batch_log_marker(self) -> None:
        """Batch stderr I/O markers are detected as storage failures."""
        batch_logs = self._batch_logs_dir()

        def run(options: Any) -> None:
            options.mode = "batch_log_marker"
            options.coordinationPath = self.coord_dir
            options.batchLogsPath = batch_logs
            slurm_recovery_workflow.main(options)

        log_text = self._run_slurm_workflow(run, retry_count=2)
        self._assert_log_contains_any(
            log_text,
            _LOG_EXCLUDE_NODES,
            "storage",
        )

    @unittest.skipUnless(
        os.environ.get("TOIL_SLURM_PARTITION_FAILOVER", "").strip(),
        "Set TOIL_SLURM_PARTITION_FAILOVER=partA,partB for failover rotation test",
    )
    def test_slurm_partition_failover_rotation(self) -> None:
        """Storage failure rotates TOIL_SLURM_PARTITION_FAILOVER for new submissions."""
        batch_logs = self._batch_logs_dir()

        def run(options: Any) -> None:
            options.mode = "worker_oserror"
            options.coordinationPath = self.coord_dir
            options.batchLogsPath = batch_logs
            slurm_recovery_workflow.main(options)

        log_text = self._run_slurm_workflow(run, retry_count=2)
        self._assert_log_contains_any(log_text, _LOG_FAILOVER)

    def test_slurm_partition_alternate_preflight(self) -> None:
        """Skip unless the target partition has Slurm Alternate= configured."""
        partition = _slurm_partition()
        if not partition:
            self.skipTest("Set TOIL_SLURM_PARTITION for partition preflight")
        if not _partition_has_alternate(partition):
            self.skipTest(
                f"Partition {partition} has no Alternate=; "
                "configure Alternate= in Slurm for in-place partition switch"
            )

    @unittest.skipUnless(
        os.environ.get(_PARTITION_SWITCH_ENV, "").strip(),
        f"Set {_PARTITION_SWITCH_ENV}=1 to run observational partition-switch test",
    )
    def test_slurm_partition_switch_observed(self) -> None:
        """
        Pass when leader log shows an in-place Alternate= switch.

        Requires the cluster to requeue a Slurm job with high Restarts while this
        test runs (see module docstring). Uses a small sort workflow.
        """
        partition = _slurm_partition()
        if not partition:
            self.skipTest("Set TOIL_SLURM_PARTITION for partition-switch test")
        if not _partition_has_alternate(partition):
            self.skipTest(f"Partition {partition} has no Alternate=")

        os.environ.setdefault("TOIL_SLURM_JOB_RESTART_THRESHOLD", "2")
        os.environ.setdefault("TOIL_SLURM_PARTITION_SWITCH_COOLDOWN", "0")
        os.environ.setdefault("TOIL_SLURM_PARTITION_SWITCH_POLL_INTERVAL", "0.25")

        job_store_locator = self._job_store_locator()
        output_file = os.path.join(self.tempDir, "sortedFile.txt")
        input_file = os.path.join(self.tempDir, "fileToSort.txt")
        options = self._base_slurm_options(job_store_locator)
        options.retryCount = 2
        options.N = defaultN
        options.outputFile = output_file
        options.fileToSort = input_file
        options.overwriteOutput = True
        options.numLines = defaultLines
        options.lineLength = defaultLineLen

        makeFileToSort(input_file, lines=defaultLines, lineLen=defaultLineLen)

        log_text = ""
        try:
            with self._leader_log_capture():
                with runMain(options):
                    pass
            log_text = self._read_leader_log()
        finally:
            subprocess.check_call(
                [resolveEntryPoint("toil"), "clean", job_store_locator]
            )

        if _LOG_PARTITION_SWITCH not in log_text:
            self.skipTest(
                "No in-place partition switch observed in leader log; "
                "re-run while Slurm requeues jobs with high Restarts on this partition"
            )
