# Copyright (c) 2016 Duke Center for Genomic and Computational Biology
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
from __future__ import annotations

import configparser
import errno
import logging
import math
import os
import shlex
import sys
from argparse import SUPPRESS, ArgumentParser, _ArgumentGroup
from collections.abc import Callable
from datetime import datetime, timedelta
from queue import Empty
import time
from typing import NamedTuple, TypeVar

import pandas

from toil.batchSystems.abstractBatchSystem import (
    EXIT_STATUS_UNAVAILABLE_VALUE,
    BatchJobExitReason,
    InsufficientSystemResources,
    UpdatedBatchJobInfo,
)
from toil.batchSystems.abstractGridEngineBatchSystem import (
    AbstractGridEngineBatchSystem,
)
from toil.batchSystems.options import OptionSetter
from toil.batchSystems.slurm_mount_recovery import (
    DEFAULT_MAX_EXCLUDED_NODES,
    DEFAULT_PARTITION_SWITCH_COOLDOWN,
    DEFAULT_PARTITION_SWITCH_POLL_INTERVAL,
    PARTITION_SWITCH_STATES,
    STORAGE_FAILURE_EXIT_CODE,
    batch_logs_indicate_storage_failure,
    env_bool,
    env_csv,
    env_float,
    env_int,
    parse_slurm_nodelist,
    parse_scontrol_job_lines,
    partition_switch_reason_matches,
    run_scontrol,
)
from toil.bus import get_job_kind
from toil.common import Config
from toil.job import JobDescription, Requirer
from toil.lib.conversions import strtobool
from toil.lib.misc import CalledProcessErrorStderr, call_command
from toil.statsAndLogging import TRACE

logger = logging.getLogger(__name__)

# (state, exit code, pending reason from scontrol/sacct)
JobStatusDetail = tuple[str | None, int | None, str | None]

# We have a complete list of Slurm states. States not in one of these aren't
# allowed. See <https://slurm.schedmd.com/squeue.html#SECTION_JOB-STATE-CODES>

# If a job is in one of these states, Slurm can't run it anymore.
# We don't include states where the job is held or paused here;
# those mean it could run and needs to wait for someone to un-hold
# it, so Toil should wait for it.
#
# We map from each terminal state to the Toil-ontology exit reason.
TERMINAL_STATES: dict[str, BatchJobExitReason] = {
    "BOOT_FAIL": BatchJobExitReason.LOST,
    "CANCELLED": BatchJobExitReason.KILLED,
    "COMPLETED": BatchJobExitReason.FINISHED,
    "DEADLINE": BatchJobExitReason.KILLED,
    "FAILED": BatchJobExitReason.FAILED,
    "NODE_FAIL": BatchJobExitReason.LOST,
    "OUT_OF_MEMORY": BatchJobExitReason.MEMLIMIT,
    "PREEMPTED": BatchJobExitReason.KILLED,
    "REVOKED": BatchJobExitReason.KILLED,
    "SPECIAL_EXIT": BatchJobExitReason.FAILED,
    "TIMEOUT": BatchJobExitReason.KILLED,
}

# If a job is in one of these states, it might eventually move to a different
# state.
NONTERMINAL_STATES: set[str] = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "RUNNING",
    "RESV_DEL_HOLD",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "REQUEUED",
    "RESIZING",
    "SIGNALING",
    "STAGE_OUT",
    "STOPPED",
    "SUSPENDED",
}


def parse_slurm_time(slurm_time: str) -> int:
    """
    Parse a Slurm-style time duration to a number of seconds.

    Slurm supports the following time formats:
    - "minutes"
    - "minutes:seconds"
    - "hours:minutes:seconds"
    - "days-hours"
    - "days-hours:minutes"
    - "days-hours:minutes:seconds"

    Raises ValueError if not parseable.
    """
    # Split on dash to check for days
    if "-" in slurm_time:
        days_str, _, slurm_time = slurm_time.partition("-")
        days = int(days_str)
    else:
        days = 0

    # Split remaining time into components and convert to integers
    time_components = [int(x) for x in slurm_time.split(":")]

    # Pad right to 3 if we have days, 2 otherwise
    time_components += [0] * ((3 if days else 2) - len(time_components))

    # Parse as base 60 from the left
    result = 0
    for component in time_components:
        result = result * 60 + component

    return days * 86400 + result


def slurm_job_number(batch_job_ref: int | str) -> int:
    """
    Normalize a Slurm batch job reference to its integer job ID.

    ``submitJob`` stores an ``int`` in ``batchJobIDs``; other call paths may
    pass ``"<job>"`` or ``"<job>.<task>"`` strings.
    """
    return int(str(batch_job_ref).split(".", 1)[0])


# For parsing user-provided option overrides (or self-generated
# options) for sbatch, we need a way to recognize long, long-with-equals, and
# short forms.
def option_detector(long: str, short: str | None = None) -> Callable[[str], bool]:
    """
    Get a function that returns true if it sees the long or short
    option.
    """

    def is_match(option: str) -> bool:
        return (
            option == f"--{long}"
            or option.startswith(f"--{long}=")
            or (short is not None and option == f"-{short}")
        )

    return is_match


def any_option_detector(options: list[str | tuple[str, str]]) -> Callable[[str], bool]:
    """
    Get a function that returns true if it sees any of the long
    options or long or short option pairs.
    """
    detectors = [
        option_detector(o) if isinstance(o, str) else option_detector(*o)
        for o in options
    ]

    def is_match(option: str) -> bool:
        for detector in detectors:
            if detector(option):
                return True
        return False

    return is_match


class SlurmBatchSystem(AbstractGridEngineBatchSystem):
    class PartitionInfo(NamedTuple):
        partition_name: str
        gres: bool
        time_limit: float
        priority: int
        cpus: str
        memory: str

    class PartitionSet:
        """
        Set of available partitions detected on the slurm batch system
        """

        default_gpu_partition: SlurmBatchSystem.PartitionInfo | None = None
        all_partitions: list[SlurmBatchSystem.PartitionInfo] | None = None
        gpu_partitions: set[str] | None = None

        def __init__(self) -> None:
            try:
                self._get_partition_info()
                self._get_gpu_partitions()
            except CalledProcessErrorStderr as e:
                logger.warning("Could not retrieve Slurm partition info due to: '%s'.", e)

        def _get_gpu_partitions(self) -> None:
            """
            Get all available GPU partitions. Also get the default GPU partition.
            :return: None
            """
            if not self.all_partitions:
                return

            gpu_partitions = [
                partition for partition in self.all_partitions if partition.gres
            ]
            self.gpu_partitions = {p.partition_name for p in gpu_partitions}
            # Grab the lowest priority GPU partition
            # If no GPU partitions are available, then set the default to None
            self.default_gpu_partition = None
            if len(gpu_partitions) > 0:
                self.default_gpu_partition = sorted(
                    gpu_partitions, key=lambda x: x.priority
                )[0]

        def _get_partition_info(self) -> None:
            """
            Call the Slurm batch system with sinfo to grab all available partitions.
            Then parse the output and store all available Slurm partitions
            :return: None
            """
            sinfo_command = ["sinfo", "-a", "-o", "%P %G %l %p %c %m"]

            sinfo = call_command(sinfo_command)

            parsed_partitions = []
            for line in sinfo.split("\n")[1:]:
                if line.strip():
                    partition_name, gres, time, priority, cpus, memory = line.split(" ")
                    try:
                        # Parse time to a number so we can compute on it
                        partition_time: float = parse_slurm_time(time)
                    except ValueError:
                        # Maybe time is unlimited?
                        partition_time = float("inf")
                    try:
                        # Parse priority to an int so we can sort on it
                        partition_priority = int(priority)
                    except ValueError:
                        logger.warning(
                            "Could not parse priority %s for partition %s, assuming high priority",
                            partition_name,
                            priority,
                        )
                        partition_priority = sys.maxsize
                    parsed_partitions.append(
                        SlurmBatchSystem.PartitionInfo(
                            partition_name.rstrip("*"),
                            gres != "(null)",
                            partition_time,
                            partition_priority,
                            cpus,
                            memory,
                        )
                    )
            self.all_partitions = parsed_partitions

        def get_partition(self, time_limit: float | None) -> str | None:
            """
            Get the partition name to use for a job with the given time limit.

            :param time_limit: Time limit in seconds.
            """

            if time_limit is None or self.all_partitions is None:
                # Just use Slurm's default
                return None

            winning_partition = None
            for partition in self.all_partitions:
                if partition.time_limit < time_limit:
                    # Can't use this
                    continue
                if winning_partition is None:
                    # Anything beats None
                    winning_partition = partition
                    continue
                if partition.gres and not winning_partition.gres:
                    # Never use a partition witn GRES if you can avoid it
                    continue
                elif not partition.gres and winning_partition.gres:
                    # Never keep a partition with GRES if we find one without
                    winning_partition = partition
                    continue
                if partition.priority > winning_partition.priority:
                    # After that, don't raise priority
                    continue
                elif partition.priority < winning_partition.priority:
                    # And always lower it
                    winning_partition = partition
                    continue
                if partition.time_limit < winning_partition.time_limit:
                    # Finally, lower time limit
                    winning_partition = partition

            # TODO: Store partitions in a better indexed way
            if winning_partition is None and len(self.all_partitions) > 0:
                # We have partitions and none of them can fit this
                raise RuntimeError(
                    f"Could not find a Slurm partition that can fit a job that runs for {time_limit} seconds"
                )

            if winning_partition is None:
                return None
            else:
                return winning_partition.partition_name

    class GridEngineThread(AbstractGridEngineBatchSystem.GridEngineThread):
        # Our boss is always the enclosing class
        boss: SlurmBatchSystem

        def getRunningJobIDs(self) -> dict[int, int]:
            # Should return a dictionary of Job IDs and number of seconds
            times = {}
            with self.runningJobsLock:
                currentjobs: dict[str, int] = {
                    str(self.batchJobIDs[x][0]): x for x in self.runningJobs
                }
            # currentjobs is a dictionary that maps a slurm job id (string) to our own internal job id
            # squeue arguments:
            # -h for no header
            # --format to get jobid i, state %t and time days-hours:minutes:seconds

            lines = call_command(
                ["squeue", "-h", "--format", "%i %t %M"], quiet=True
            ).split("\n")
            for line in lines:
                values = line.split()
                if len(values) < 3:
                    continue
                slurm_jobid, state, elapsed_time = values
                if slurm_jobid in currentjobs and state == "R":
                    try:
                        seconds_running = parse_slurm_time(elapsed_time)
                    except ValueError:
                        # slurm may return INVALID instead of a time
                        seconds_running = 0
                    times[currentjobs[slurm_jobid]] = seconds_running

            return times

        def killJob(self, jobID: int) -> None:
            call_command(["scancel", self.getBatchSystemID(jobID)])

        def _slurm_restart_threshold(self) -> int:
            return env_int("TOIL_SLURM_JOB_RESTART_THRESHOLD", 5)

        def checkOnJobs(self) -> bool:
            if self.boss.partition_switch_watch:
                interval = env_float(
                    "TOIL_SLURM_PARTITION_SWITCH_POLL_INTERVAL",
                    DEFAULT_PARTITION_SWITCH_POLL_INTERVAL,
                )
                now = time.time()
                if now - self.boss._partition_switch_last_poll >= interval:
                    self.boss._partition_switch_last_poll = now
                    self._poll_partition_switch_watch()
                self._checkOnJobsTimestamp = None
            return super().checkOnJobs()

        def _poll_partition_switch_watch(self) -> None:
            threshold = self._slurm_restart_threshold()
            for slurm_job_id in list(self.boss.partition_switch_watch):
                try:
                    stdout = run_scontrol("show", "job", str(slurm_job_id), quiet=True)
                except (CalledProcessErrorStderr, OSError) as e:
                    logger.debug(
                        "Partition switch watch: could not query job %s: %s",
                        slurm_job_id,
                        e,
                    )
                    continue
                for record in stdout.strip().split("\n\n"):
                    job_details = self._parse_scontrol_show_job_record(record)
                    if not job_details:
                        continue
                    self._track_partition_switch_watch(slurm_job_id, job_details, threshold)
                    self._maybe_invoke_partition_switch(slurm_job_id, job_details, threshold)
                    state = self._canonicalize_state(job_details.get("JobState", ""))
                    if state in TERMINAL_STATES:
                        self.boss.partition_switch_watch.discard(slurm_job_id)

        @staticmethod
        def _parse_scontrol_show_job_record(record: str) -> dict[str, str]:
            lines: list[str] = []
            for line in record.splitlines():
                lines.extend(line.split())
            if not lines:
                return {}
            return parse_scontrol_job_lines(lines)

        def _track_partition_switch_watch(
            self, slurm_job_id: int, job_details: dict[str, str], threshold: int
        ) -> None:
            try:
                restarts = int(job_details.get("Restarts", 0))
            except ValueError:
                restarts = 0
            if restarts >= max(0, threshold - 1):
                self.boss.partition_switch_watch.add(slurm_job_id)

        def _nodes_from_job_details(self, job_details: dict[str, str]) -> set[str]:
            nodes: set[str] = set()
            for key in ("NodeList", "BatchHost", "AllocNode"):
                nodes.update(parse_slurm_nodelist(job_details.get(key)))
            return nodes

        def _terminal_job_storage_failure(
            self,
            toil_job_id: int,
            slurm_job_id: int,
            status: JobStatusDetail,
            job_details: dict[str, str] | None,
        ) -> bool:
            state, rc, _reason = status
            if rc == STORAGE_FAILURE_EXIT_CODE:
                return True
            if job_details:
                nodes = self._nodes_from_job_details(job_details)
                if nodes and batch_logs_indicate_storage_failure(
                    self.boss, toil_job_id, slurm_job_id
                ):
                    return True
            return batch_logs_indicate_storage_failure(
                self.boss, toil_job_id, slurm_job_id
            )

        def _apply_storage_failure_outcome(
            self,
            toil_job_id: int,
            slurm_job_id: int,
            job_details: dict[str, str] | None,
        ) -> tuple[int, BatchJobExitReason]:
            nodes: set[str] = set()
            if job_details:
                nodes = self._nodes_from_job_details(job_details)
            if not nodes:
                logger.warning(
                    "Storage I/O failure for Toil job %s (Slurm job %s) but no node "
                    "names from Slurm job details; sbatch --exclude and drain will "
                    "not target a host for this incident",
                    toil_job_id,
                    slurm_job_id,
                )
            self.boss.on_storage_failure(nodes)
            self.boss._lost_job_first_seen.pop(slurm_job_id, None)
            return (STORAGE_FAILURE_EXIT_CODE, BatchJobExitReason.STORAGE)

        def prepareSubmission(
            self,
            cpu: int,
            memory: int,
            jobID: int,
            command: str,
            jobName: str,
            job_environment: dict[str, str] | None = None,
            gpus: int | None = None,
            usePreferredPartition: bool | None = True,
            comment: str | None = "",
        ) -> list[str]:
            # Make sure to use exec so we can get Slurm's signals in the Toil
            # worker instead of having an intervening Bash
            return self.prepareSbatch(
                cpu, memory, jobID, jobName, job_environment, gpus, usePreferredPartition, comment
            ) + [f"--wrap=exec {command}"]

        def submitJob(self, subLine: list[str]) -> str:
            try:
                # Slurm is not quite clever enough to follow the XDG spec on
                # its own. If the submission command sees e.g. XDG_RUNTIME_DIR
                # in our environment, it will send it along (especially with
                # --export=ALL), even though it makes a promise to the job that
                # Slurm isn't going to keep. It also has a tendency to create
                # /run/user/<uid> *at the start* of a job, but *not* keep it
                # around for the duration of the job.
                #
                # So we hide the whole XDG universe from Slurm before we make
                # the submission.
                # Might as well hide DBUS also.
                # This doesn't get us a trustworthy XDG session in Slurm, but
                # it does let us see the one Slurm tries to give us.
                no_session_environment = os.environ.copy()
                session_names = [
                    n
                    for n in no_session_environment.keys()
                    if n.startswith("XDG_") or n.startswith("DBUS_")
                ]
                for name in session_names:
                    del no_session_environment[name]

                output = call_command(subLine, env=no_session_environment)
                # sbatch prints a line like 'Submitted batch job 2954103'
                result = int(output.strip().split()[-1])
                logger.info("sbatch submitted job %d", result)
                return str(result)
            except OSError as e:
                logger.error(f"sbatch command failed with error: {e}")
                raise e

        def coalesce_job_exit_codes(
            self, batch_job_id_list: list[str]
        ) -> list[int | tuple[int, BatchJobExitReason | None] | None]:
            """
            Collect all job exit codes in a single call.

            :param batch_job_id_list: list of Job ID strings, where each string
                has the form ``<job>[.<task>]``.

            :return: list of job exit codes or exit code, exit reason pairs
                associated with the list of job IDs.

            :raises CalledProcessErrorStderr: if communicating with Slurm went
                wrong.

            :raises OSError: if job details are not available because a Slurm
                command could not start.
            """
            logger.log(
                TRACE, "Getting exit codes for slurm jobs: %s", batch_job_id_list
            )
            # Convert batch_job_id_list to list of integer job IDs.
            job_id_list = [int(id.split(".")[0]) for id in batch_job_id_list]
            status_dict = self._get_job_details(job_id_list)
            exit_codes: list[int | tuple[int, BatchJobExitReason | None] | None] = []
            with self.runningJobsLock:
                slurm_to_toil = {
                    slurm_job_number(self.batchJobIDs[x][0]): x
                    for x in self.runningJobs
                    if x in self.batchJobIDs
                }
            for slurm_job_id, status in status_dict.items():
                toil_job_id = slurm_to_toil.get(slurm_job_id)
                code = self._get_job_return_code(
                    status, slurm_job_id=slurm_job_id, toil_job_id=toil_job_id
                )
                if toil_job_id is not None:
                    code = self._finalize_exit_code(
                        code, toil_job_id, slurm_job_id, status
                    )
                exit_codes.append(code)
            return exit_codes

        def getJobExitCode(
            self, batchJobID: str
        ) -> int | tuple[int, BatchJobExitReason | None] | None:
            """
            Get job exit code for given batch job ID.
            :param batchJobID: string of the form "<job>[.<task>]".
            :return: integer job exit code.
            """
            logger.log(TRACE, "Getting exit code for slurm job: %s", batchJobID)
            # Convert batchJobID to an integer job ID.
            job_id = slurm_job_number(batchJobID)
            status_dict = self._get_job_details([job_id])
            status = status_dict[job_id]
            toil_job_id = None
            with self.runningJobsLock:
                for running_id in self.runningJobs:
                    if (
                        running_id in self.batchJobIDs
                        and slurm_job_number(self.batchJobIDs[running_id][0])
                        == job_id
                    ):
                        toil_job_id = running_id
                        break
            code = self._get_job_return_code(
                status, slurm_job_id=job_id, toil_job_id=toil_job_id
            )
            if toil_job_id is not None:
                code = self._finalize_exit_code(code, toil_job_id, job_id, status)
            return code

        def getUpdatedBatchJob(self, maxWait):
            try:
                logger.debug("getUpdatedBatchJob: Job updates")
                item = self.updatedJobsQueue.get(timeout=maxWait)
                self.updatedJobsQueue.task_done()
                jobID, retcode, exit_reason = (
                    self.jobIDs[item.jobID],
                    item.exitStatus,
                    item.exit_reason,
                )
                self.currentjobs -= {self.jobIDs[item.jobID]}
            except Empty:
                logger.debug("getUpdatedBatchJob: Job queue is empty")
            else:
                return UpdatedBatchJobInfo(
                    jobID=jobID,
                    exitStatus=retcode,
                    wallTime=None,
                    exitReason=exit_reason,
                )

        def _get_job_details(
            self, job_id_list: list[int]
        ) -> dict[int, JobStatusDetail]:
            """
            Helper function for `getJobExitCode` and `coalesce_job_exit_codes`.
            Fetch job details from Slurm's accounting system or job control system.
            :param job_id_list: list of integer Job IDs.
            :return: dict of job statuses, where key is the integer job ID, and
                value is a tuple containing the job's state and exit code.
            :raises CalledProcessErrorStderr: if communicating with Slurm went
                wrong.
            :raises OSError: if job details are not available because a Slurm
                command could not start.
            """

            status_dict = {}
            scontrol_problem: Exception | None = None

            try:
                # Get all the job details we can from scontrol, which we think
                # might be faster/less dangerous than sacct searching, even
                # though it can't be aimed at more than one job.
                status_dict.update(self._getJobDetailsFromScontrol(job_id_list))
            except (CalledProcessErrorStderr, OSError) as e:
                if isinstance(e, OSError):
                    logger.warning("Could not run scontrol: %s", e)
                else:
                    logger.warning("Error from scontrol: %s", e)
                scontrol_problem = e

            logger.debug("After scontrol, got statuses: %s", status_dict)

            # See what's not handy in scontrol (or everything if we couldn't
            # call it).
            sacct_job_id_list = self._remaining_jobs(job_id_list, status_dict)

            logger.debug("Remaining jobs to find out about: %s", sacct_job_id_list)

            try:
                # Ask sacct about those jobs
                status_dict.update(self._getJobDetailsFromSacct(sacct_job_id_list))
            except (CalledProcessErrorStderr, OSError) as e:
                if isinstance(e, OSError):
                    logger.warning("Could not run sacct: %s", e)
                else:
                    logger.warning("Error from sacct: %s", e)
                if scontrol_problem is not None:
                    # Neither approach worked at all
                    raise

            # One of the methods worked, so we have at least (None, None, None)
            # values filled in for all jobs.
            assert len(status_dict) == len(job_id_list)

            return status_dict

        def _get_job_return_code(
            self,
            status: JobStatusDetail,
            slurm_job_id: int | None = None,
            toil_job_id: int | None = None,
        ) -> int | tuple[int, BatchJobExitReason | None] | None:
            """
            Given a Slurm return code, status pair, summarize them into a Toil return code, exit reason pair.

            The return code may have already been OR'd with the 128-offset
            Slurm-reported signal.

            Slurm will report return codes of 0 even if jobs time out instead
            of succeeding:

                2093597|TIMEOUT|0:0
                2093597.batch|CANCELLED|0:15

            So we guarantee here that, if the Slurm status string is not a
            successful one as defined in
            <https://slurm.schedmd.com/squeue.html#SECTION_JOB-STATE-CODES>, we
            will not return a successful return code.

            Helper function for `getJobExitCode` and `coalesce_job_exit_codes`.
            :param status: tuple containing the job's state, return code, and reason from Slurm.
            :return: the job's return code for Toil if it's completed, otherwise None.
            """
            state, rc, reason = status

            if state not in TERMINAL_STATES:
                if reason == "BadConstraints":
                    logger.warning("[SlurmJobHandler] Bad Constrains reason detected.")
                    return BatchJobExitReason.BADCONSTRAINTS
                # Don't treat the job as exited yet
                return None

            exit_reason = TERMINAL_STATES[state]

            if exit_reason == BatchJobExitReason.FINISHED:
                # The only state that should produce a 0 ever is COMPLETED. So
                # if the job is COMPLETED and the exit reason is thus FINISHED,
                # pass along the code it has.
                return (rc, exit_reason)  # type: ignore[return-value] # mypy doesn't understand enums well

            if exit_reason == BatchJobExitReason.LOST:
                lost_timeout = env_float("TOIL_SLURM_LOST_JOB_TIMEOUT", None)
                if lost_timeout is not None and slurm_job_id is not None:
                    now = time.time()
                    first_seen = self.boss._lost_job_first_seen.get(slurm_job_id)
                    if first_seen is None:
                        self.boss._lost_job_first_seen[slurm_job_id] = now
                        logger.debug(
                            "Slurm job %s in LOST-related state %s; waiting up to %ss for Slurm",
                            slurm_job_id,
                            state,
                            lost_timeout,
                        )
                        return None
                    if now - first_seen < lost_timeout:
                        return None
                    logger.warning(
                        "Slurm job %s remained in LOST-related state %s for %ss; "
                        "treating as failed so Toil can retry",
                        slurm_job_id,
                        state,
                        lost_timeout,
                    )
                    self.boss._lost_job_first_seen.pop(slurm_job_id, None)
                    return (EXIT_STATUS_UNAVAILABLE_VALUE, BatchJobExitReason.LOST)
                return None

            if slurm_job_id is not None:
                self.boss._lost_job_first_seen.pop(slurm_job_id, None)

            if rc == STORAGE_FAILURE_EXIT_CODE:
                return (STORAGE_FAILURE_EXIT_CODE, BatchJobExitReason.STORAGE)

            if rc == 0:
                # The job claims to be in a state other than COMPLETED, but
                # also to have not encountered a problem. Say the exit status
                # is unavailable.
                return (EXIT_STATUS_UNAVAILABLE_VALUE, exit_reason)
            # If the code is nonzero, pass it along.
            return (rc, exit_reason)  # type: ignore[return-value] # mypy doesn't understand enums well

        def _finalize_exit_code(
            self,
            code: int | tuple[int, BatchJobExitReason | None] | None,
            toil_job_id: int,
            slurm_job_id: int,
            status: JobStatusDetail,
        ) -> int | tuple[int, BatchJobExitReason | None] | None:
            if code is None:
                return None
            if isinstance(code, BatchJobExitReason):
                return code
            if isinstance(code, int):
                exit_code, exit_reason = code, None
            else:
                exit_code, exit_reason = code
            if exit_code == STORAGE_FAILURE_EXIT_CODE:
                job_details = self._fetch_scontrol_job_details(slurm_job_id)
                return self._apply_storage_failure_outcome(
                    toil_job_id, slurm_job_id, job_details
                )
            job_details = self._fetch_scontrol_job_details(slurm_job_id)
            if self._terminal_job_storage_failure(
                toil_job_id, slurm_job_id, status, job_details
            ):
                return self._apply_storage_failure_outcome(
                    toil_job_id, slurm_job_id, job_details
                )
            if isinstance(code, int):
                return code
            return (exit_code, exit_reason)

        def _fetch_scontrol_job_details(self, slurm_job_id: int) -> dict[str, str] | None:
            try:
                stdout = run_scontrol("show", "job", str(slurm_job_id), quiet=True)
            except (CalledProcessErrorStderr, OSError):
                return None
            records = stdout.strip().split("\n\n")
            if not records:
                return None
            return self._parse_scontrol_show_job_record(records[0])

        def get_last_partition_switch_details(self, comment):
            """Get last partition switch time if comment contains it and the switch was done before 2 min

            :param comment: Job comment
            :type comment: str

            :return: Last partition switch time
            :rtype: tuple
            """
            # A job might contain user comments at times.
            # We would like to store swtich count and time.
            # Format: <user_comments>;ToilSlurmPartitionSwtich:<switch_count>+<swtich_time>
            last_switch_time, switch_count = (None, None)
            if comment:
                if comment.__contains__("ToilSlurmPartitionSwitch"):
                    switch_details = (
                        next(filter(lambda x: x.startswith("Toil"), comment.split(";")))
                        .split(":")[1]
                        .split("+")
                    )
                    switch_count = int(switch_details[0])
                    last_switch_time = int(switch_details[1])
                    updated_comment = f"ToilSlurmPartitionSwitch:{switch_count + 1}+{int(time.time())}"
                else:
                    updated_comment = (
                        f"{comment};ToilSlurmPartitionSwitch:1+{int(time.time())}"
                    )
            else:
                updated_comment = f"ToilSlurmPartitionSwitch:1+{int(time.time())}"
            logger.debug("Updated Comment: %s", updated_comment)
            return (last_switch_time, switch_count, updated_comment)

        def _get_alternate_partition(self, partition: str | None) -> str | None:
            if not partition:
                return None
            try:
                stdout = run_scontrol("show", "partition", partition, quiet=True)
            except (CalledProcessErrorStderr, OSError) as e:
                logger.debug("Could not read partition %s: %s", partition, e)
                return None
            alternate: str | None = None
            for line in stdout.splitlines():
                for item in line.split():
                    if item.startswith("Alternate="):
                        alternate = item.split("=", 1)[1]
                        break
                if alternate:
                    break
            if not alternate:
                logger.debug(
                    "Cannot switch partition: no Alternate= configured for %s",
                    partition,
                )
                return None
            try:
                alt_stdout = run_scontrol("show", "partition", alternate, quiet=True)
            except (CalledProcessErrorStderr, OSError) as e:
                logger.debug(
                    "Cannot switch partition: could not read alternate %s: %s",
                    alternate,
                    e,
                )
                return None
            alternate_state: str | None = None
            for line in alt_stdout.splitlines():
                for item in line.split():
                    if item.startswith("State="):
                        alternate_state = item.split("=", 1)[1]
                        break
                if alternate_state:
                    break
            if alternate_state and alternate_state != "UP":
                logger.debug(
                    "Cannot switch partition: alternate partition %s is %s",
                    alternate,
                    alternate_state,
                )
                return None
            return alternate

        def _maybe_invoke_partition_switch(
            self,
            slurm_job_id: int,
            job_details: dict[str, str],
            restart_threshold: int,
        ) -> None:
            state = self._canonicalize_state(job_details.get("JobState", ""))
            reason = job_details.get("Reason")
            if state not in PARTITION_SWITCH_STATES:
                return
            if not partition_switch_reason_matches(reason):
                return
            try:
                restart_count = int(job_details.get("Restarts", 0))
            except ValueError:
                restart_count = 0
            self._track_partition_switch_watch(slurm_job_id, job_details, restart_threshold)
            if restart_count < restart_threshold:
                return
            partition = job_details.get("Partition")
            alternate_partition = self._get_alternate_partition(partition)
            if not alternate_partition:
                return
            if partition == alternate_partition:
                return
            comment = job_details.get("Comment")
            last_switch_time, switch_count, updated_comment = (
                self.get_last_partition_switch_details(comment)
            )
            cooldown = env_int(
                "TOIL_SLURM_PARTITION_SWITCH_COOLDOWN",
                DEFAULT_PARTITION_SWITCH_COOLDOWN,
            )
            if last_switch_time and (int(time.time()) - last_switch_time) < cooldown:
                logger.debug(
                    "Skipping partition switch for job %s; last switch was %ss ago",
                    slurm_job_id,
                    int(time.time()) - last_switch_time,
                )
                return
            if switch_count:
                restart_threshold *= switch_count + 1
                if restart_count < restart_threshold:
                    return
            self._switch_job_partition(
                slurm_job_id, alternate_partition, updated_comment, state
            )

        def _switch_job_partition(
            self,
            slurm_job_id: int,
            alternate_partition: str,
            updated_comment: str,
            state: str,
        ) -> None:
            update_args = [
                "update",
                f"JobId={slurm_job_id}",
                f"Partition={alternate_partition}",
                f'Comment={updated_comment}',
            ]
            try:
                if state == "RUNNING":
                    run_scontrol("hold", str(slurm_job_id), quiet=True)
                    run_scontrol(*update_args, quiet=True)
                    run_scontrol("release", str(slurm_job_id), quiet=True)
                else:
                    run_scontrol(*update_args, quiet=True)
                logger.info(
                    "Job %s switched to alternate partition %s (was %s)",
                    slurm_job_id,
                    alternate_partition,
                    state,
                )
                self.boss.partition_switch_watch.discard(slurm_job_id)
            except (CalledProcessErrorStderr, OSError) as e:
                logger.warning(
                    "Job %s could not switch to partition %s (%s); canceling for Toil retry",
                    slurm_job_id,
                    alternate_partition,
                    e,
                )
                try:
                    call_command(["scancel", str(slurm_job_id)])
                except (CalledProcessErrorStderr, OSError) as cancel_err:
                    logger.warning(
                        "Could not cancel job %s after failed partition switch: %s",
                        slurm_job_id,
                        cancel_err,
                    )

        def check_and_change_partition(
            self, job_details: dict[str, str], restart_threshold: int | None = None
        ) -> None:
            """Switch a Slurm-requeued job to its partition's Alternate= target when restarts exceed threshold."""
            if restart_threshold is None:
                restart_threshold = self._slurm_restart_threshold()
            job_id_raw = job_details.get("JobId")
            if job_id_raw is None:
                return
            slurm_job_id = slurm_job_number(job_id_raw)
            self._maybe_invoke_partition_switch(
                slurm_job_id, job_details, restart_threshold
            )

        def _canonicalize_state(self, state: str) -> str:
            """
            Turn a state string form SLURM into just the state token like "CANCELED".
            """

            # Slurm will sometimes send something like "CANCELED by 30065" in
            # the state column for some reason.

            state_token = state

            if " " in state_token:
                state_token = state.split(" ", 1)[0]

            if (
                state_token not in TERMINAL_STATES
                and state_token not in NONTERMINAL_STATES
            ):
                raise RuntimeError("Toil job in unimplemented Slurm state " + state)

            return state_token

        def _remaining_jobs(
            self,
            job_id_list: list[int],
            job_details: dict[int, JobStatusDetail],
        ) -> list[int]:
            """
            Given a list of job IDs and a list of job details (state, exit
            code, reason), get the list of job IDs where the details are
            (None, None, None) (or are missing).
            """
            return [
                j
                for j in job_id_list
                if job_details.get(j, (None, None, None)) == (None, None, None)
            ]

        def _getJobDetailsFromSacct(
            self,
            job_id_list: list[int],
        ) -> dict[int, JobStatusDetail]:
            """
            Get SLURM job exit codes for the jobs in `job_id_list` by running `sacct`.

            Handles querying manageable time periods until all jobs have information.

            There is no guarantee of inter-job consistency: one job may really
            finish after another, but we might see the earlier-finishing job
            still running and the later-finishing job finished.

            :param job_id_list: list of integer batch job IDs.
            :return: dict of job statuses, where key is the job-id, and value
                is a tuple containing the job's state, exit code, and reason.
                Jobs with no information reported from Slurm will have
                (None, None, None).
            """

            # Pick a now
            now = datetime.now().astimezone(None)
            # Decide when to start the search (first copy of past midnight)
            begin_time = now.replace(hour=0, minute=0, second=0, microsecond=0, fold=0)
            # And when to end (a day after that)
            end_time = begin_time + timedelta(days=1)
            while end_time < now:
                # If something goes really weird, advance up to our chosen now
                end_time += timedelta(days=1)
            # If we don't go around the loop at least once, we might end up
            # with an empty dict being returned, which shouldn't happen. We
            # need the (None, None, None) entries for jobs we can't find.
            assert end_time >= self.boss.start_time

            results: dict[int, JobStatusDetail] = {}

            while len(job_id_list) > 0 and end_time >= self.boss.start_time:
                # There are still jobs to look for and our search isn't
                # exclusively for stuff that only existed before our workflow
                # started.
                results.update(
                    self._get_job_details_from_sacct_for_range(
                        job_id_list, begin_time, end_time
                    )
                )
                job_id_list = self._remaining_jobs(job_id_list, results)
                # If we have to search again, search the previous day. But
                # overlap a tiny bit so the endpoints don't exactly match, in
                # case Slurm is not working with inclusive intervals.
                # TODO: is Slurm working with inclusive intervals?
                end_time = begin_time + timedelta(seconds=1)
                begin_time = end_time - timedelta(days=1, seconds=1)

            if end_time < self.boss.start_time and len(job_id_list) > 0:
                # This is suspicious.
                logger.warning(
                    "Could not find any information from sacct after "
                    "workflow start at %s about jobs: %s",
                    self.boss.start_time.isoformat(),
                    job_id_list,
                )

            return results

        def _get_job_details_from_sacct_for_range(
            self,
            job_id_list: list[int],
            begin_time: datetime,
            end_time: datetime,
        ) -> dict[int, JobStatusDetail]:
            """
            Get SLURM job exit codes for the jobs in `job_id_list` by running `sacct`.

            Internally, Slurm's accounting thinks in wall clock time, so for
            efficiency you need to only search relevant real-time periods.

            :param job_id_list: list of integer batch job IDs.
            :param begin_time: An aware datetime of the earliest time to search
            :param end_time: An aware datetime of the latest time to search
            :return: dict of job statuses, where key is the job-id, and value
                is a tuple containing the job's state, exit code, and reason.
                Jobs with no information reported from Slurm will have
                (None, None, None).
            """

            assert begin_time.tzinfo is not None, "begin_time must be aware"
            assert end_time.tzinfo is not None, "end_time must be aware"

            def stringify(t: datetime) -> str:
                """
                Convert an aware time local time, and format it *without* a
                trailing time zone indicator.
                """
                # TODO: What happens when we get an aware time that's ambiguous
                # in local time? Or when the local timezone changes while we're
                # sending things to Slurm or doing a progressive search back?
                naive_t = t.astimezone(None).replace(tzinfo=None)
                return naive_t.isoformat(timespec="seconds")

            job_ids = ",".join(str(id) for id in job_id_list)
            args = [
                "sacct",
                "-n",  # no header
                "-j",
                job_ids,  # job
                "--format",
                "JobIDRaw,State,ExitCode",  # specify output columns
                "-P",  # separate columns with pipes
                "-S",
                stringify(begin_time),
                "-E",
                stringify(end_time),
            ]

            # Collect the job statuses in a dict; key is the job-id, value is a tuple containing
            # job state and exit status. Initialize dict before processing output of `sacct`.
            job_statuses: dict[int, JobStatusDetail] = {}

            try:
                stdout = call_command(args, quiet=True)
            except OSError as e:
                if e.errno == errno.E2BIG:
                    # Argument list is too big, recurse on half the argument list
                    if len(job_id_list) == 1:
                        # 1 is too big, we can't recurse further, bail out
                        raise
                    job_statuses.update(
                        self._get_job_details_from_sacct_for_range(
                            job_id_list[: len(job_id_list) // 2],
                            begin_time,
                            end_time,
                        )
                    )
                    job_statuses.update(
                        self._get_job_details_from_sacct_for_range(
                            job_id_list[len(job_id_list) // 2 :],
                            begin_time,
                            end_time,
                        )
                    )
                    return job_statuses
                else:
                    raise

            for job_id in job_id_list:
                job_statuses[job_id] = (None, None, None)

            for line in stdout.splitlines():
                values = line.strip().split("|")
                if len(values) < 3:
                    continue
                state: str
                job_id_raw, state, exitcode = values
                state = self._canonicalize_state(state)
                logger.log(
                    TRACE, "%s state of job %s is %s", args[0], job_id_raw, state
                )
                # JobIDRaw is in the form JobID[.JobStep]; we're not interested in job steps.
                job_id_parts = job_id_raw.split(".")
                if len(job_id_parts) > 1:
                    continue
                job_id = int(job_id_parts[0])
                status: int
                signal: int
                status, signal = (int(n) for n in exitcode.split(":"))
                reason = ""
                if state in PARTITION_SWITCH_STATES:
                    job_details = self._fetch_scontrol_job_details(job_id) or {}
                    reason = job_details.get("Reason", "")
                    if reason == "BadConstraints":
                        status = 7
                    logger.debug(
                        "Job %s in %s (reason=%s); checking Slurm partition switch",
                        job_id,
                        state,
                        reason,
                    )
                    self.check_and_change_partition(job_details)
                if signal > 0:
                    # A non-zero signal may indicate e.g. an out-of-memory killed job
                    status = 128 + signal
                logger.log(
                    TRACE,
                    "%s exit code of job %d is %s, return status %d",
                    args[0],
                    job_id,
                    exitcode,
                    status,
                )
                job_statuses[job_id] = state, status, reason
            logger.log(TRACE, "%s returning job statuses: %s", args[0], job_statuses)
            return job_statuses

        def _getJobDetailsFromScontrol(
            self, job_id_list: list[int]
        ) -> dict[int, JobStatusDetail]:
            """
            Get SLURM job exit codes for the jobs in `job_id_list` by running `scontrol`.
            :param job_id_list: list of integer batch job IDs.
            :return: dict of job statuses, where key is the job-id, and value is a tuple
            containing the job's state and exit code.
            """
            args = ["scontrol", "show", "job"]
            # `scontrol` can only return information about a single job,
            # or all the jobs it knows about.
            if len(job_id_list) == 1:
                args.append(str(job_id_list[0]))

            stdout = call_command(args, quiet=True)

            # Job records are separated by a blank line.
            job_records = None
            if isinstance(stdout, str):
                job_records = stdout.strip().split("\n\n")
            elif isinstance(stdout, bytes):
                job_records = stdout.decode("utf-8").strip().split("\n\n")

            # Collect the job statuses in a dict; key is the job-id, value is a tuple containing
            # job state and exit status. Initialize dict before processing output of `scontrol`.
            job_statuses: dict[int, JobStatusDetail] = {}
            job_id: int | None
            for job_id in job_id_list:
                job_statuses[job_id] = (None, None, None)

            # `scontrol` will report "No jobs in the system", if there are no jobs in the system,
            # and if no job-id was passed as argument to `scontrol`.
            if len(job_records) > 0 and job_records[0] == "No jobs in the system":
                return job_statuses

            for record in job_records:
                job: dict[str, str] = {}
                job_id = None
                for line in record.splitlines():
                    for item in line.split():
                        # Output is in the form of many key=value pairs, multiple pairs on each line
                        # and multiple lines in the output. Each pair is pulled out of each line and
                        # added to a dictionary.
                        # Note: In some cases, the value itself may contain white-space. So, if we find
                        # a key without a value, we consider that key part of the previous value.
                        bits = item.split("=", 1)
                        if len(bits) == 1:
                            job[key] += " " + bits[0]  # type: ignore[has-type]  # we depend on the previous iteration to populate key
                        else:
                            key = bits[0]
                            job[key] = bits[1]
                    # The first line of the record contains the JobId. Stop processing the remainder
                    # of this record, if we're not interested in this job.
                    job_id = int(job["JobId"])
                    if job_id not in job_id_list:
                        logger.log(
                            TRACE, "%s job %d is not in the list", args[0], job_id
                        )
                        break
                if job_id is None or job_id not in job_id_list:
                    continue
                state = job["JobState"]
                state = self._canonicalize_state(state)
                reason = ""
                if state in PARTITION_SWITCH_STATES:
                    reason = job.get("Reason", "")
                    if reason == "BadConstraints":
                        job["ExitCode"] = 7
                    logger.debug(
                        "Job %s in %s (reason=%s); checking Slurm partition switch",
                        job_id,
                        state,
                        reason,
                    )
                    self.check_and_change_partition(job)
                logger.log(TRACE, "%s state of job %s is %s", args[0], job_id, state)
                try:
                    exitcode = job["ExitCode"]
                    if exitcode is not None:
                        status, signal = (int(n) for n in exitcode.split(":"))
                        if signal > 0:
                            # A non-zero signal may indicate e.g. an out-of-memory killed job
                            status = 128 + signal
                        logger.log(
                            TRACE,
                            "%s exit code of job %d is %s, return status %d",
                            args[0],
                            job_id,
                            exitcode,
                            status,
                        )
                        rc = status
                    else:
                        rc = None
                except KeyError:
                    rc = None
                job_statuses[job_id] = (state, rc, reason)
            logger.log(TRACE, "%s returning job statuses: %s", args[0], job_statuses)
            return job_statuses

        ###
        ### Implementation-specific helper methods
        ###
        def select_partition(self, cpus, mem, accelerators, preferred=True):
            """Select suitable slurm partition based on requirements. Checks state of partition.

            :param cpus: required cps
            :type cpus: int
            :param mem: required memory
            :type mem: integer
            :param preferred: respect partition preference, defaults to True
            :type preferred: bool, optional
            :return: suitable partition for the requirements
            :rtype: str
            """
            gpu = True if accelerators else False
            logger.info("GPU Required: %s", gpu)
            # Intentionally we check here if gpu nodes exist and choose them if required.
            # This is because we need an approach to ignore accelerator specification if gpu partition not found.
            usable_resources = self.batchSystemResources
            logger.info(
                "Detected Cluster Partitions: %s", usable_resources.partitions.unique()
            )
            if gpu:
                if not any(self.batchSystemResources["gpu"]):
                    logger.warning("""
                    Ignoring specified accelerator requirements, as there are no gpu nodes available in the cluster.
                    """)
                else:
                    usable_resources = self.batchSystemResources[
                        self.batchSystemResources["gpu"]
                    ]
            else:
                usable_resources = self.batchSystemResources[
                    ~self.batchSystemResources["gpu"]
                ]
            if "preference" in self.batchSystemResources.columns:
                possible_partitions = usable_resources.partitions[
                    (self.batchSystemResources["cputot"] >= cpus)
                    & (self.batchSystemResources["realmemory"] >= mem)
                    & (self.batchSystemResources["preference"] == preferred)
                ].values
            else:
                possible_partitions = usable_resources.partitions[
                    (self.batchSystemResources["cputot"] >= cpus)
                    & (self.batchSystemResources["realmemory"] >= mem)
                ].values
            if len(possible_partitions) != 0:
                usable_partitions = []
                logger.info("Feasible Partitions: %s", possible_partitions)
                for partition in possible_partitions:
                    partition_state = (
                        os.popen(
                            f"""
                        scontrol -o show partition {partition} |
                        sed 's/ /\\n/g' |
                        grep State |
                        cut -d "=" -f2
                        """
                        )
                        .read()
                        .strip()
                    )
                    if partition_state == "UP":
                        usable_partitions.append(partition)
                    else:
                        logger.info(
                            "Skipping partition: %s, due to state being %s",
                            partition,
                            partition_state,
                        )
                logger.info("Selectable Partitions: %s", usable_partitions)
                return usable_partitions[0]

            if "preference" in self.batchSystemResources.columns:
                logger.warning(
                    "Could not find a partition to suffice cpus: %s, memory: %s, accelerators: %s and preferred type: %s",
                    cpus,
                    mem,
                    accelerators,
                    preferred,
                )
                logger.info("Trying with preferred type: %s", not preferred)
                return self.select_partition(cpus, mem, accelerators, not preferred)
            else:
                logger.error(
                    "Could not find a partition to suffice cpus: %s, memory: %s, accelerators: %s",
                    cpus,
                    mem,
                    accelerators,
                )
                return

        def prepareSbatch(
            self,
            cpu: int,
            mem: int,
            jobID: int,
            jobName: str,
            job_environment: dict[str, str] | None,
            gpus: int | None,
            usePreferredPartition: Optional[bool],
            comment: Optional[str],
        ) -> list[str]:
            """
            Returns the sbatch command line to run to queue the job.
            """
            timeout = os.getenv("TOIL_SLURM_JOB_TIMEOUT", "02:30:00")
            # Start by naming the job
            sbatch_line = ["sbatch", "-t", timeout, "-J", f"toil_job_{jobID}_{jobName}"]

            # Make sure the job gets a signal before it disappears so that e.g.
            # container cleanup finally blocks can run. Ask for SIGINT so we
            # can get the default Python KeyboardInterrupt which third-party
            # code is likely to plan for. Make sure to send it to the batch
            # shell process with "B:", not to all the srun steps it launches
            # (because there shouldn't be any). We cunningly replaced the batch
            # shell process with the Toil worker process, so Toil should be
            # able to get the signal.
            #
            # TODO: Add a way to detect when the job failed because it
            # responded to this signal and use the right exit reason for it.
            sbatch_line.append("--signal=B:INT@30")
            if gpus:
                sbatch_line = sbatch_line[:1] + [f"--gres=gpu:{gpus}"] + sbatch_line[1:]
            environment = {}
            environment.update(self.boss.environment)
            if job_environment:
                environment.update(job_environment)

            # "Native extensions" for SLURM (see DRMAA or SAGA)
            # Also any extra arguments from --slurmArgs or TOIL_SLURM_ARGS
            nativeConfig: str = self.boss.config.slurm_args  # type: ignore[attr-defined]

            is_any_mem_option = any_option_detector(
                ["mem", "mem-per-cpu", "mem-per-gpu"]
            )
            is_any_cpus_option = any_option_detector(
                [("cpus-per-task", "c"), "cpus-per-gpu"]
            )
            is_export_option = option_detector("export")
            is_export_file_option = option_detector("export-file")
            is_time_option = option_detector("time", "t")
            is_partition_option = option_detector("partition", "p")

            # We will fill these in with stuff parsed from TOIL_SLURM_ARGS, or
            # with our own determinations if they aren't there.

            # --export=[ALL,]<environment_toil_variables>
            export_all = True
            export_list = []  # Some items here may be multiple comma-separated values
            time_limit: int | None = self.boss.config.slurm_time  # type: ignore[attr-defined]
            partition: str | None = None

            if nativeConfig is not None:
                logger.debug(
                    "Native SLURM options appended to sbatch: %s", nativeConfig
                )

                # Do a mini argument parse to pull out export and parse time if
                # needed
                args = shlex.split(nativeConfig)
                i = 0
                while i < len(args):
                    arg = args[i]
                    if is_any_mem_option(arg) or is_any_cpus_option(arg):
                        # Prohibit arguments that set CPUs or memory
                        raise ValueError(
                            f"Cannot use Slurm argument {arg} which conflicts "
                            f"with Toil's own arguments to Slurm"
                        )
                    elif is_export_option(arg):
                        # Capture the export argument value so we can modify it
                        export_all = False
                        if "=" not in arg:
                            if i + 1 >= len(args):
                                raise ValueError(
                                    f"No value supplied for Slurm {arg} argument"
                                )
                            i += 1
                            export_list.append(args[i])
                        else:
                            export_list.append(arg.split("=", 1)[1])
                    elif is_export_file_option(arg):
                        # Keep --export-file but turn off --export=ALL in that
                        # case.
                        export_all = False
                        sbatch_line.append(arg)
                    elif is_time_option(arg):
                        # Capture the time limit in seconds so we can use it for picking a partition
                        if "=" not in arg:
                            if i + 1 >= len(args):
                                raise ValueError(
                                    f"No value supplied for Slurm {arg} argument"
                                )
                            i += 1
                            time_string = args[i]
                        else:
                            time_string = arg.split("=", 1)[1]
                        time_limit = parse_slurm_time(time_string)
                    elif is_partition_option(arg):
                        # Capture the partition so we can run checks on it and know not to assign one
                        if "=" not in arg:
                            if i + 1 >= len(args):
                                raise ValueError(
                                    f"No value supplied for Slurm {arg} argument"
                                )
                            i += 1
                            partition = args[i]
                        else:
                            partition = arg.split("=", 1)[1]
                    else:
                        # Other arguments pass through.
                        sbatch_line.append(arg)
                    i += 1

            if export_all:
                # We don't have any export overrides so we ened to start with
                # an ALL
                export_list.append("ALL")

            if environment:
                argList = []

                for k, v in environment.items():
                    # TODO: The sbatch man page doesn't say we can quote these;
                    # if we need to send characters like , itself we need to
                    # use --export-file and clean it up when the command has
                    # been issued.
                    quoted_value = shlex.quote(os.environ[k] if v is None else v)
                    argList.append(f"{k}={quoted_value}")

                export_list.extend(argList)

            # If partition isn't set and we have a GPU partition override
            # that applies, apply it
            gpu_partition_override: str | None = self.boss.config.slurm_gpu_partition  # type: ignore[attr-defined]
            if partition is None and gpus and gpu_partition_override:
                partition = gpu_partition_override

            # If partition isn't set and we have a parallel partition override
            # that applies, apply it
            parallel_env: str | None = self.boss.config.slurm_pe  # type: ignore[attr-defined]
            if partition is None and cpu and cpu > 1 and parallel_env:
                partition = parallel_env

            # If partition isn't set and we have a general partition override
            # that applies, apply it
            partition_override: str | None = self.boss.config.slurm_partition  # type: ignore[attr-defined]
            if partition is None and partition_override:
                partition = partition_override

            if partition is None and gpus:
                # Send to a GPU partition
                gpu_partition = self.boss.partitions.default_gpu_partition
                if gpu_partition is None:
                    # no gpu partitions are available, raise an error
                    raise RuntimeError(
                        f"The job {jobName} is requesting GPUs, but the Slurm cluster does not appear to have an accessible partition with GPUs"
                    )
                if time_limit is not None and gpu_partition.time_limit < time_limit:
                    # TODO: find the lowest-priority GPU partition that has at least each job's time limit!
                    logger.warning(
                        "Trying to submit a job that needs %s seconds to partition %s that has a limit of %s seconds",
                        time_limit,
                        gpu_partition.partition_name,
                        gpu_partition.time_limit,
                    )
                partition = gpu_partition.partition_name

            if partition is None:
                # Pick a partition based on time limit
                partition = self.boss.partitions.get_partition(time_limit)

            if self.boss.active_failover_partition:
                partition = self.boss.active_failover_partition

            # Now generate all the arguments
            if len(export_list) > 0:
                # add --export to the sbatch
                sbatch_line.append("--export=" + ",".join(export_list))
            if partition is not None:
                sbatch_line.append(f"--partition={partition}")
            if self.boss.excluded_nodes:
                max_excluded = env_int(
                    "TOIL_SLURM_MAX_EXCLUDED_NODES", DEFAULT_MAX_EXCLUDED_NODES
                )
                exclude_list = ",".join(
                    sorted(self.boss.excluded_nodes)[:max_excluded]
                )
                if exclude_list:
                    sbatch_line.append(f"--exclude={exclude_list}")
            if gpus:
                # Generate GPU assignment argument
                sbatch_line.append(f"--gres=gpu:{gpus}")
                if self.boss.partitions.gpu_partitions is None:
                    logger.warning(
                        f"Job {jobName} needs GPUs, but specified partition {partition} might not have them. This job may not work."
                        f"Try specifying a different partition"
                    )
                elif (
                    partition is not None
                    and partition not in self.boss.partitions.gpu_partitions
                ):
                    # the specified partition is not compatible, so warn the user that the job may not work
                    logger.warning(
                        f"Job {jobName} needs GPUs, but specified partition {partition} does not have them. This job may not work."
                        f"Try specifying one of these partitions instead: {', '.join(self.boss.partitions.gpu_partitions)}."
                    )
            if mem is not None and self.boss.config.slurm_allocate_mem:  # type: ignore[attr-defined]
                # memory passed in is in bytes, but slurm expects megabytes
                slurm_mem = math.ceil(mem / 2**20)
                sbatch_line.append(f"--mem={slurm_mem}")
            else:
                slurm_mem = None
            if cpu is not None:
                slurm_cpu = math.ceil(cpu)
                sbatch_line.append(f"--cpus-per-task={slurm_cpu}")
            if time_limit is not None:
                # Put all the seconds in the seconds slot
                sbatch_line.append(f"--time=0:{time_limit}")

            if slurm_mem and slurm_cpu:
                partition = self.select_partition(
                    slurm_cpu,
                    slurm_mem,
                    accelerators=gpus,
                    preferred=usePreferredPartition,
                )
                logger.info(
                    "Selected partition: %s based on cpus: %s and memory: %s of preferred type: %s",
                    partition,
                    slurm_cpu,
                    slurm_mem,
                    usePreferredPartition,
                )
                sbatch_line.append(f"--partition={partition}")
            else:
                logger.info(
                    "Skipping slurm partition selection as mem and cpu are not specified."
                )

            if comment is not None:
                sbatch_line.append(f"--comment={comment}")

            stdoutfile: str = self.boss.format_std_out_err_path(jobID, "%j", "out")
            stderrfile: str = self.boss.format_std_out_err_path(jobID, "%j", "err")
            sbatch_line.extend(["-o", stdoutfile, "-e", stderrfile])
            return sbatch_line

    def __init__(
        self, config: Config, maxCores: float, maxMemory: float, maxDisk: float
    ) -> None:
        # Background thread starts in super().__init__ and may call checkOnJobs
        # immediately; set partition-switch state before that.
        self.partition_switch_watch: set[int] = set()
        self._lost_job_first_seen: dict[int, float] = {}
        self._partition_switch_last_poll = 0.0
        super().__init__(config, maxCores, maxMemory, maxDisk)
        self.partitions = SlurmBatchSystem.PartitionSet()
        # Record when the workflow started, so we know when to stop looking for
        # jobs we ran.
        self.start_time = datetime.now().astimezone(None)
        self.excluded_nodes: set[str] = set()
        failover = getattr(config, "slurm_partition_failover", None)
        if isinstance(failover, str):
            self.failover_partitions = [
                p.strip() for p in failover.split(",") if p.strip()
            ]
        elif failover:
            self.failover_partitions = list(failover)
        else:
            self.failover_partitions = env_csv("TOIL_SLURM_PARTITION_FAILOVER")
        self.failover_partition_index = 0
        self.active_failover_partition: str | None = None

    def advance_failover_partition(self) -> str | None:
        """Rotate to the next configured failover partition for new sbatch submissions."""
        if not self.failover_partitions:
            return None
        if self.active_failover_partition is None:
            self.failover_partition_index = 0
            self.active_failover_partition = self.failover_partitions[0]
        else:
            self.failover_partition_index = (
                self.failover_partition_index + 1
            ) % len(self.failover_partitions)
            self.active_failover_partition = self.failover_partitions[
                self.failover_partition_index
            ]
        logger.info(
            "Slurm partition failover: subsequent worker jobs will use partition %s",
            self.active_failover_partition,
        )
        return self.active_failover_partition

    def record_storage_failure_nodes(self, nodes: set[str]) -> None:
        if not nodes:
            return
        max_nodes = env_int("TOIL_SLURM_MAX_EXCLUDED_NODES", DEFAULT_MAX_EXCLUDED_NODES)
        added: list[str] = []
        skipped_cap: list[str] = []
        for node in nodes:
            if node in self.excluded_nodes:
                continue
            if len(self.excluded_nodes) >= max_nodes:
                skipped_cap.append(node)
                continue
            self.excluded_nodes.add(node)
            added.append(node)
        if skipped_cap:
            logger.warning(
                "Cannot add %s to sbatch --exclude: cap TOIL_SLURM_MAX_EXCLUDED_NODES=%s "
                "(%s nodes already excluded). Enable --slurmDrainBadNodes for persistently "
                "bad nodes on large clusters.",
                ", ".join(sorted(skipped_cap)),
                max_nodes,
                len(self.excluded_nodes),
            )
        if added:
            logger.warning(
                "Excluding Slurm nodes after storage I/O failure: %s",
                ", ".join(sorted(added)),
            )

    def drain_nodes_if_enabled(self, nodes: set[str]) -> None:
        drain = env_bool("TOIL_SLURM_DRAIN_BAD_NODES") or bool(
            getattr(self.config, "slurm_drain_bad_nodes", False)
        )
        if not drain:
            return
        reason = "Toil: mount I/O failure"
        for node in sorted(nodes):
            try:
                info = run_scontrol("show", "node", node, quiet=True)
            except (CalledProcessErrorStderr, OSError) as e:
                logger.warning("Could not inspect node %s before drain: %s", node, e)
                continue
            if "State=IDLE" not in info and "State=ALLOCATED" not in info and (
                "DRAIN" in info or "DOWN" in info
            ):
                logger.debug("Node %s already drained or down", node)
                continue
            try:
                run_scontrol(
                    "update",
                    f"NodeName={node}",
                    "State=DRAIN",
                    f"Reason={reason}",
                    quiet=True,
                )
                logger.info("Drained Slurm node %s (%s)", node, reason)
            except (CalledProcessErrorStderr, OSError) as e:
                logger.warning("Could not drain Slurm node %s: %s", node, e)

    def on_storage_failure(self, nodes: set[str]) -> None:
        self.record_storage_failure_nodes(nodes)
        self.advance_failover_partition()
        self.drain_nodes_if_enabled(nodes)

    # Override issuing jobs so we can check if we need to use Slurm's magic
    # whole-node-memory feature.
    def issueBatchJob(
        self,
        command: str,
        job_desc: JobDescription,
        job_environment: dict[str, str] | None = None,
    ) -> int:
        # Avoid submitting internal jobs to the batch queue, handle locally
        local_id = self.handleLocalJob(command, job_desc)
        if local_id is not None:
            return local_id
        else:
            self.check_resource_request(job_desc)
            gpus = self.count_needed_gpus(job_desc)
            job_id = self.getNextJobID()
            self.currentJobs.add(job_id)

            if "memory" not in job_desc.requirements and self.config.slurm_default_all_mem:  # type: ignore[attr-defined]
                # The job doesn't have its own memory requirement, and we are
                # defaulting to whole node memory. Use Slurm's 0-memory sentinel.
                memory = 0
            else:
                # Use the memory actually on the job, or the Toil default memory
                memory = job_desc.memory

            self.newJobsQueue.put(
                (
                    job_id,
                    job_desc.cores,
                    memory,
                    command,
                    get_job_kind(job_desc.get_names()),
                    job_environment,
                    gpus,
                    job_desc.usePreferredPartition,
                    job_desc.comment,
                )
            )
            logger.debug(
                "Issued the job command: %s with job id: %s and job name %s on spot capacity: %s with comment %s",
                command,
                str(job_id),
                get_job_kind(job_desc.get_names()),
                job_desc.usePreferredPartition,
                job_desc.comment
            )
        return job_id

    def _check_accelerator_request(self, requirer: Requirer) -> None:
        for accelerator in requirer.accelerators:
            if accelerator["kind"] != "gpu":
                raise InsufficientSystemResources(
                    requirer,
                    "accelerators",
                    details=[
                        f"The accelerator {accelerator} could not be provided"
                        "The Toil Slurm batch system only supports gpu accelerators at the moment."
                    ],
                )

    ###
    ### The interface for SLURM
    ###
    @classmethod
    def _check_accelerator_request(self, requirer: Requirer) -> None:
        for accelerator in requirer.accelerators:
            if accelerator["kind"] != "gpu":
                # We can only provide GPUs, and of those only nvidia ones.
                raise InsufficientSystemResources(
                    requirer,
                    "accelerators",
                    details=[
                        f"The accelerator {accelerator} could not be provided.",
                        "Slurm can only provide gpu accelerators.",
                    ],
                )
            # if not any(self.batchSystemResources['gputot'] >= accelerator['count']):
            #     raise InsufficientSystemResources(requirer, 'accelerators', details=[
            #         f'The requested number of accelerators {accelerator} could not be provided.',
            #         f'Slurm cluster currently has {self.batchSystemResources["gputot"]}.'
            #     ])

    @classmethod
    def assessBatchResources(cls):
        slurm_partition_configs = (
            os.popen(
                r"""
            scontrol show node -o |
            sed 's/\[.*//g' |
            sed 's/NodeName=/\[/' |
            sed 's/ /\] /' |
            sed 's/ \+/\n/g' |
            sed 's/(null)//g' |
            egrep "=|\["
            """
            )
            .read()
            .strip()
        )
        # print(slurm_partition_configs)
        config = configparser.ConfigParser()
        config.read_string(slurm_partition_configs)
        config_dicts = []
        for section, val in config._sections.items():
            cdict = {"NodeName": section}
            for k, v in val.items():
                cdict[k] = v
            config_dicts.append(cdict)

        config_data = pandas.DataFrame.from_dict(config_dicts)
        # print(config_data)
        req_configs = config_data[
            [
                "partitions",
                "cputot",
                "realmemory",
                "gres",
            ]
        ]
        slurm_resources = req_configs.groupby("partitions").max().reset_index()
        slurm_resources["gputot"] = slurm_resources.gres.apply(
            lambda x: int(x.split(":")[2]) if x else None
        )
        slurm_resources["gpu"] = ~slurm_resources.gputot.isnull()
        slurm_resources.drop(columns="gres", inplace=True)
        preference = os.getenv("TOIL_SLURM_PARTITON_PREFERED")
        if preference:
            logger.info("Setting slurm partition preference: %s", preference)
            slurm_resources["preference"] = slurm_resources.partitions.str.contains(
                preference, case=False
            )
        else:
            logger.info("No slurm partition preference set")
        slurm_resources[["cputot", "realmemory"]] = slurm_resources[
            ["cputot", "realmemory"]
        ].astype(int)
        slurm_resources.sort_values(["cputot", "realmemory"], inplace=True)
        return slurm_resources

    # `scontrol show config` can get us the slurm config, and there are values
    # SchedulerTimeSlice and AcctGatherNodeFreq in there, but
    # SchedulerTimeSlice is for time-sharing preemtion and AcctGatherNodeFreq
    # is for reporting resource statistics (and can be 0). Slurm does not
    # actually seem to have a scheduling granularity or tick rate. So we don't
    # implement getWaitDuration().

    @classmethod
    def add_options(cls, parser: ArgumentParser | _ArgumentGroup) -> None:

        parser.add_argument(
            "--slurmAllocateMem",
            dest="slurm_allocate_mem",
            type=strtobool,
            default=True,
            env_var="TOIL_SLURM_ALLOCATE_MEM",
            help="If False, do not use --mem. Used as a workaround for Slurm clusters that reject jobs "
            "with memory allocations.",
        )
        # Keep these deprcated options for backward compatibility
        parser.add_argument(
            "--dont_allocate_mem",
            action="store_false",
            dest="slurm_allocate_mem",
            help=SUPPRESS,
        )
        parser.add_argument(
            "--allocate_mem",
            action="store_true",
            dest="slurm_allocate_mem",
            help=SUPPRESS,
        )

        parser.add_argument(
            "--slurmDefaultAllMem",
            dest="slurm_default_all_mem",
            type=strtobool,
            default=False,
            env_var="TOIL_SLURM_DEFAULT_ALL_MEM",
            help="If True, assign Toil jobs without their own memory requirements all available "
            "memory on a Slurm node (via Slurm --mem=0).",
        )
        parser.add_argument(
            "--slurmTime",
            dest="slurm_time",
            type=parse_slurm_time,
            default=None,
            env_var="TOIL_SLURM_TIME",
            help="Slurm job time limit, in [DD-]HH:MM:SS format.",
        )
        parser.add_argument(
            "--slurmPartition",
            dest="slurm_partition",
            default=None,
            env_var="TOIL_SLURM_PARTITION",
            help="Partition to send Slurm jobs to.",
        )
        parser.add_argument(
            "--slurmGPUPartition",
            dest="slurm_gpu_partition",
            default=None,
            env_var="TOIL_SLURM_GPU_PARTITION",
            help="Partition to send Slurm jobs to if they ask for GPUs.",
        )
        parser.add_argument(
            "--slurmPE",
            dest="slurm_pe",
            default=None,
            env_var="TOIL_SLURM_PE",
            help="Special partition to send Slurm jobs to if they ask for more than 1 CPU.",
        )
        parser.add_argument(
            "--slurmArgs",
            dest="slurm_args",
            default="",
            env_var="TOIL_SLURM_ARGS",
            help="Extra arguments to pass to Slurm.",
        )
        parser.add_argument(
            "--slurmPartitionFailover",
            dest="slurm_partition_failover",
            default=None,
            env_var="TOIL_SLURM_PARTITION_FAILOVER",
            help="Comma-separated Slurm partitions to rotate through after storage I/O failures "
            "(applies to newly submitted worker jobs).",
        )
        parser.add_argument(
            "--slurmDrainBadNodes",
            dest="slurm_drain_bad_nodes",
            type=strtobool,
            default=False,
            env_var="TOIL_SLURM_DRAIN_BAD_NODES",
            help="If True, drain Slurm nodes where a worker reported mount/storage I/O failure "
            "(requires permission to run scontrol update on nodes).",
        )

    OptionType = TypeVar("OptionType")

    @classmethod
    def setOptions(cls, setOption: OptionSetter) -> None:
        setOption("slurm_allocate_mem")
        setOption("slurm_default_all_mem")
        setOption("slurm_time")
        setOption("slurm_partition")
        setOption("slurm_gpu_partition")
        setOption("slurm_pe")
        setOption("slurm_args")
        setOption("slurm_partition_failover")
        setOption("slurm_drain_bad_nodes")
