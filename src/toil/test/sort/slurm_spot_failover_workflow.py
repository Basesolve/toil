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
Manual / integrative probe for Slurm spot failover before --doubleMem.

Expected sequence on a ParallelCluster with ``*-cs`` Alternate=``*-cod``:

1. Attempts on ``*cs`` exit 143 -> exit reason ``SPOT``, memory unchanged.
2. After ``--slurmSpotFailoverStrikes`` (default 2), next sbatch uses ``*cod``.
3. First attempt on ``*cod`` also exits 143 -> exit reason ``FAILED``,
   ``--doubleMem`` doubles memory.
4. Retry on ``*cod`` with doubled memory succeeds.

Run on a Slurm head node (after deploying the patched Toil), for example::

    export TOIL_SLURM_PARTITON_PREFERED=cs
    export TOIL_SLURM_SPOT_FAILOVER=1
    export TOIL_SLURM_SPOT_FAILOVER_STRIKES=2
    export TOIL_SLURM_JOB_TIMEOUT='02:30:00'
    export TOIL_CHECK_ENV=True
    /path/to/.venv/bin/python -m toil.test.sort.slurm_spot_failover_workflow \\
        /augmet-mp/job_stores/spot_doublemem_probe
"""
from __future__ import annotations

import os
import socket

from toil.job import Job
from toil.realtimeLogger import RealtimeLogger

# Keep small so *cs / *cod nodes can schedule; doubling must stay visible in logs.
_BASE_MEMORY = "2G"
_BASE_MEMORY_BYTES = 2 * 1024**3


def _partition_family(hostname: str) -> str:
    """Return the leading Slurm-style token (e.g. lcs, lcod, mcs)."""
    return hostname.split("-", 1)[0].lower()


def _is_spot_host(hostname: str) -> bool:
    family = _partition_family(hostname)
    return family.endswith("cs") and not family.endswith("cod")


def _is_ondemand_host(hostname: str) -> bool:
    return _partition_family(hostname).endswith("cod")


def probe_spot_then_double_mem(job: Job) -> str:
    """
    Fail with 143 on spot hosts; on on-demand hosts fail once at base memory
    then succeed after --doubleMem.
    """
    hostname = socket.gethostname()
    memory = int(job.memory)
    memory_gi = memory / (1024**3)
    RealtimeLogger.info(
        "[spot/doubleMem probe] host=%s family=%s memory=%.2fGi (%s bytes)",
        hostname,
        _partition_family(hostname),
        memory_gi,
        memory,
    )

    if _is_spot_host(hostname):
        RealtimeLogger.info(
            "[spot/doubleMem probe] on spot host %s; exiting 143 (expect SPOT, "
            "memory unchanged until failover)",
            hostname,
        )
        os._exit(143)

    if _is_ondemand_host(hostname):
        if memory <= _BASE_MEMORY_BYTES:
            RealtimeLogger.info(
                "[spot/doubleMem probe] on on-demand host %s with base memory; "
                "exiting 143 (expect FAILED + doubleMem)",
                hostname,
            )
            os._exit(143)
        RealtimeLogger.info(
            "[spot/doubleMem probe] on on-demand host %s with doubled memory "
            "%.2fGi; success",
            hostname,
            memory,
        )
        return f"ok:{hostname}:{memory}"

    RealtimeLogger.info(
        "[spot/doubleMem probe] unexpected host %s (neither *cs nor *cod); "
        "failing so the schedule is visible in the leader log",
        hostname,
    )
    raise RuntimeError(
        f"Probe landed on unexpected host {hostname}; "
        "expected *cs then *cod after spot failover"
    )


def main() -> None:
    parser = Job.Runner.getDefaultArgumentParser()
    options = parser.parse_args()
    options.batchSystem = "slurm"
    options.caching = False
    options.disableProgress = True
    options.realTimeLogging = True
    options.doubleMem = True
    options.clean = "always"
    # options.slurm_spot_failover = True
    # if getattr(options, "slurm_spot_failover_strikes", None) is None:
    #     options.slurm_spot_failover_strikes = 2
    # if getattr(options, "slurm_spot_failover_cooldown", None) is None:
    #     options.slurm_spot_failover_cooldown = 0
    # if options.retryCount is None or options.retryCount < 5:
    #     options.retryCount = 5
    # Config expects bytes; CLI parsing converts "2G", but post-parse assigns do not.
    options.defaultMemory = _BASE_MEMORY_BYTES

    root = Job.wrapJobFn(
        probe_spot_then_double_mem,
        memory=_BASE_MEMORY,
        disk="2G",
        cores=1,
    )
    Job.Runner.startToil(root, options)


if __name__ == "__main__":
    main()
