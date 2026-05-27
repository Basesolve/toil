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
Minimal Toil workflow for Slurm mount-recovery integration tests.

Used by :mod:`toil.test.sort.slurmSortTest` to simulate storage I/O failures on
a real Slurm cluster.
"""
from __future__ import annotations

import errno
import os
import sys

from configargparse import ArgumentParser

from toil.batchSystems.slurm_mount_recovery import STORAGE_FAILURE_EXIT_CODE
from toil.common import Toil
from toil.job import Job
from toil.realtimeLogger import RealtimeLogger

# Marker file name under the shared batch-logs directory (visible on all nodes).
_STORAGE_SIM_MARKER = "toil_storage_sim_attempted"

_JOB_MEMORY = "512M"


def setup_storage_failure_test(
    job: Job, mode: str, coordination_path: str, batch_logs_dir: str, options
) -> str:
    """Dispatch a single child that simulates one storage failure then succeeds on retry."""
    return job.addChildJobFn(
        storage_failure_child,
        mode,
        coordination_path,
        batch_logs_dir,
        options=options,
        memory=_JOB_MEMORY,
    ).rv()


def storage_failure_child(
    job: Job, mode: str, coordination_path: str, batch_logs_dir: str, options
) -> str:
    """
    Fail once with a storage-like error, then return on retry.

    Uses a marker file in ``batch_logs_dir`` so a retried job on another node
    does not fail again.
    """
    marker_path = os.path.join(batch_logs_dir, _STORAGE_SIM_MARKER)
    if os.path.exists(marker_path):
        RealtimeLogger.info("Storage failure simulation: retry succeeded")
        return "recovered"

    with open(marker_path, "w", encoding="utf-8") as handle:
        handle.write("1")

    RealtimeLogger.info("Storage failure simulation: mode=%s", mode)
    if mode == "worker_oserror":
        probe = os.path.join(coordination_path, "toil_test_io_probe")
        try:
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("x")
        except OSError:
            pass
        raise OSError(
            errno.EIO,
            "Input/output error",
            probe,
        )
    if mode == "batch_log_marker":
        print("Input/output error", file=sys.stderr)
        sys.exit(STORAGE_FAILURE_EXIT_CODE)
    raise RuntimeError(f"Unknown storage failure simulation mode: {mode}")


def main(options=None) -> None:
    if options is None:
        parser = ArgumentParser()
        Job.Runner.addToilOptions(parser)
        parser.add_argument(
            "--mode",
            required=True,
            choices=("worker_oserror", "batch_log_marker"),
            help="How the child job simulates storage failure",
        )
        parser.add_argument(
            "--coordinationPath",
            required=True,
            help="Coordination directory path (must match --coordinationDir)",
        )
        parser.add_argument(
            "--batchLogsPath",
            required=True,
            help="Batch logs directory path (must match --batchLogsDir)",
        )
        options = parser.parse_args()

    coordination_path = os.path.abspath(options.coordinationPath)
    batch_logs_path = os.path.abspath(options.batchLogsPath)
    os.makedirs(coordination_path, exist_ok=True)
    os.makedirs(batch_logs_path, exist_ok=True)

    marker_path = os.path.join(batch_logs_path, _STORAGE_SIM_MARKER)
    if os.path.exists(marker_path):
        os.remove(marker_path)

    with Toil(options) as workflow:
        if not workflow.options.restart:
            result = workflow.start(
                Job.wrapJobFn(
                    setup_storage_failure_test,
                    options.mode,
                    coordination_path,
                    batch_logs_path,
                    options=options,
                    memory=_JOB_MEMORY,
                )
            )
        else:
            result = workflow.restart()
        assert result == "recovered"


if __name__ == "__main__":
    main()
