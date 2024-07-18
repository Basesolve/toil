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
import logging
import math
import os
from argparse import ArgumentParser, _ArgumentGroup
from queue import Empty
from shlex import quote
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union

# =======
import time
import configparser
import pandas as pd
from toil.batchSystems.abstractBatchSystem import (
    BatchJobExitReason,
    InsufficientSystemResources,
    UpdatedBatchJobInfo,
    EXIT_STATUS_UNAVAILABLE_VALUE,
)

# =======
from toil.batchSystems.abstractGridEngineBatchSystem import (
    AbstractGridEngineBatchSystem,
)
from toil.batchSystems.options import OptionSetter
from toil.job import Requirer
from toil.lib.misc import CalledProcessErrorStderr, call_command

logger = logging.getLogger(__name__)

# We have a complete list of Slurm states. States not in one of these aren't
# allowed. See <https://slurm.schedmd.com/squeue.html#SECTION_JOB-STATE-CODES>

# If a job is in one of these states, Slurm can't run it anymore.
# We don't include states where the job is held or paused here;
# those mean it could run and needs to wait for someone to un-hold
# it, so Toil should wait for it.
#
# We map from each terminal state to the Toil-ontology exit reason.
TERMINAL_STATES: Dict[str, BatchJobExitReason] = {
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
    "TIMEOUT": BatchJobExitReason.KILLED
}

# If a job is in one of these states, it might eventually move to a different
# state.
NONTERMINAL_STATES: Set[str] = {
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
    "SUSPENDED"
} 

class SlurmBatchSystem(AbstractGridEngineBatchSystem):

    class GridEngineThread(AbstractGridEngineBatchSystem.GridEngineThread):

        def getRunningJobIDs(self):
            # Should return a dictionary of Job IDs and number of seconds
            times = {}
            with self.runningJobsLock:
                currentjobs = {str(self.batchJobIDs[x][0]): x for x in self.runningJobs}
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
                    seconds_running = self.parse_elapsed(elapsed_time)
                    times[currentjobs[slurm_jobid]] = seconds_running

            return times

        def killJob(self, jobID):
            call_command(["scancel", self.getBatchSystemID(jobID)])

        def prepareSubmission(
            self,
            cpu: int,
            memory: int,
            jobID: int,
            command: str,
            jobName: str,
            job_environment: Optional[Dict[str, str]] = None,
            gpus: Optional[int] = None,
            usePreferredPartition: Optional[bool] = True,
            comment: Optional[str] = None,
        ) -> List[str]:
            return self.prepareSbatch(
                cpu,
                memory,
                jobID,
                jobName,
                job_environment,
                gpus,
                usePreferredPartition,
                comment,
            ) + [f"--wrap=exec {command}"]

        def submitJob(self, subLine):
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
                return result
            except OSError as e:
                logger.error(f"sbatch command failed with error: {e}")
                raise e

        def coalesce_job_exit_codes(
            self, batch_job_id_list: list
        ) -> List[Union[int, Tuple[int, Optional[BatchJobExitReason]], None]]:
            """
            Collect all job exit codes in a single call.
            :param batch_job_id_list: list of Job ID strings, where each string has the form
            "<job>[.<task>]".
            :return: list of job exit codes or exit code, exit reason pairs associated with the list of job IDs.
            """
            logger.debug("Getting exit codes for slurm jobs: %s", batch_job_id_list)
            # Convert batch_job_id_list to list of integer job IDs.
            job_id_list = [int(id.split(".")[0]) for id in batch_job_id_list]
            status_dict = self._get_job_details(job_id_list)
            exit_codes = []
            for _, status in status_dict.items():
                exit_codes.append(self._get_job_return_code(status))
            return exit_codes

        def getJobExitCode(
            self, batchJobID: str
        ) -> Union[int, Tuple[int, Optional[BatchJobExitReason]], None]:
            """
            Get job exit code for given batch job ID.
            :param batchJobID: string of the form "<job>[.<task>]".
            :return: integer job exit code.
            """
            logger.debug("Getting exit code for slurm job: %s", batchJobID)
            # Convert batchJobID to an integer job ID.
            job_id = int(batchJobID.split(".")[0])
            status_dict = self._get_job_details([job_id])
            status = status_dict[job_id]
            return self._get_job_return_code(status)

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

        def _get_job_details(self, job_id_list: list) -> dict:
            """
            Helper function for `getJobExitCode` and `coalesce_job_exit_codes`.
            Fetch job details from Slurm's accounting system or job control system.
            :param job_id_list: list of integer Job IDs.
            :return: dict of job statuses, where key is the integer job ID, and value is a tuple
            containing the job's state and exit code.
            """
            try:
                status_dict = self._getJobDetailsFromSacct(job_id_list)
            except CalledProcessErrorStderr:
                status_dict = self._getJobDetailsFromScontrol(job_id_list)
            return status_dict

        def _get_job_return_code(
            self, status: tuple
        ) -> Union[int, Tuple[int, Optional[BatchJobExitReason]], None]:
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
            :param status: tuple containing the job's state and it's return code from Slurm.
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
                return (rc, exit_reason)

            if exit_reason == BatchJobExitReason.LOST:
                # logger.debug(
                #     "[SlurmJobHandler] NODE_FAIL encountered. Waiting for slurm to use other nodes in partition."
                # )
                return None

            if rc == 0:
                # The job claims to be in a state other than COMPLETED, but
                # also to have not encountered a problem. Say the exit status
                # is unavailable.
                return (EXIT_STATUS_UNAVAILABLE_VALUE, exit_reason)
            # If the code is nonzero, pass it along.
            return (rc, exit_reason)  # If the code is nonzero, pass it along.

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

        def check_and_change_partition(self, job_details, restart_threshold=5):
            """Get the job restart count and switch partition.
            TODO: restart_threshold=-1 implies node count per partition.

            :param job_id: pending jobs id
            :type job_id: int
            :param restart_threshold: number of restarts to wait for before updating partition, defaults to 5
            :type restart_threshold: int, optional
            """
            # logger.debug("Slurm job details: %s", job_details)
            # comment would not be available if not definied during submission
            job_id = job_details.get("JobId")
            comment = job_details.get("Comment")
            restart_count = int(job_details.get("Restarts"))
            partition = job_details.get("Partition")
            alternate_partition = (
                os.popen(
                    f"""
                scontrol -o show partition {partition} |
                sed 's/ /\\n/g' |
                grep 'Alternate' |
                cut -d "=" -f2
                """
                )
                .read()
                .strip()
            )
            if alternate_partition:
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
                if partition_state != "UP":
                    logger.debug(
                        "Cannot switch partition: Configured alternate partition %s is %s",
                        alternate_partition,
                        partition_state,
                    )
                    return
            else:
                logger.debug(
                    "Cannot switch partition: No alternate partition configured for %s",
                    partition,
                )
                return

            # set max_possible restart threshold
            # if restart_threshold == -1:
            #     restart_threshold = total_nodes
            last_switch_time, switch_count, updated_comment = (
                self.get_last_partition_switch_details(comment)
            )
            if last_switch_time:
                if (int(time.time()) - last_switch_time) < 300000:
                    logger.debug(
                        "Seems like last patition switch happened just 5 min before. Skipping switch for now"
                    )
                    return
                restart_threshold *= switch_count + 1
                logger.debug(
                    "Partition was already switched for the job, doubling the restart threshold to %s",
                    restart_threshold,
                )
            if int(restart_count) >= restart_threshold:
                # make sure comment is not none and contains the partition switch term
                logger.info(
                    "Job %s seems to have restarted by slurm beyond the threshold %s. Switching to alternate partition %s",
                    job_id,
                    restart_threshold,
                    alternate_partition,
                )
                switch_command = f'scontrol update jobid={job_id} partition={alternate_partition} comment="{updated_comment}"'
                logger.info(f"Executing: {switch_command}")
                switch_exit_code = os.system(switch_command)
                if switch_exit_code == 0:
                    logger.info(
                        "Job: %s has been swithced to alternate partition: %s",
                        job_id,
                        alternate_partition,
                    )
                else:
                    logger.warning(
                        "Job: %s could not be swithced to alternate partition: %s, Error Code: %s",
                        job_id,
                        alternate_partition,
                        switch_exit_code,
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

            if state_token not in TERMINAL_STATES and state_token not in NONTERMINAL_STATES:
                raise RuntimeError("Toil job in unimplemented Slurm state " + state)
            
            return state_token

        def _getJobDetailsFromSacct(self, job_id_list: list) -> dict:
            """
            Get SLURM job exit codes for the jobs in `job_id_list` by running `sacct`.
            :param job_id_list: list of integer batch job IDs.
            :return: dict of job statuses, where key is the job-id, and value is a tuple
            containing the job's state and exit code.
            """
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
                "1970-01-01",
            ]  # override start time limit
            stdout = call_command(args, quiet=True)

            # Collect the job statuses in a dict; key is the job-id, value is a tuple containing
            # job state and exit status. Initialize dict before processing output of `sacct`.
            job_statuses = {}
            for job_id in job_id_list:
                job_statuses[job_id] = (None, None, None)

            for line in stdout.splitlines():
                values = line.strip().split("|")
                if len(values) < 3:
                    continue
                job_id_raw, state, exitcode = values
                state = self._canonicalize_state(state)
                logger.debug("%s state of job %s is %s", args[0], job_id_raw, state)
                # JobIDRaw is in the form JobID[.JobStep]; we're not interested in job steps.
                job_id_parts = job_id_raw.split(".")
                if len(job_id_parts) > 1:
                    continue
                job_id = int(job_id_parts[0])
                status, signal = (int(n) for n in exitcode.split(":"))
                reason = ""
                if state == "PENDING":
                    # sacct does not report the job pending reason in realtime. but scontrol does.
                    job_details = (
                        os.popen(
                            f"""
                        scontrol -o show job {job_id} |
                        sed 's/ /\\n/g'
                        """
                        )
                        .read()
                        .strip()
                        .splitlines()
                    )
                    # 'Reason|Comment|Restarts|Partition'
                    jdict = {}
                    for item in job_details:
                        bits = item.split("=", 1)
                        if len(bits) == 1:
                            jdict[key] += " " + bits[0]
                        else:
                            key = bits[0]
                            jdict[key] = bits[1]
                    reason = jdict.get("Reason")
                    if reason == "BadConstraints":
                        status = 7
                    if reason == "BeginTime":
                        logger.debug(
                            "Job: %s is in %s state. Checking if alternate partition to be used.",
                            job_id,
                            state,
                        )
                        user_slurm_restart_thresh = int(
                            os.getenv("TOIL_SLURM_JOB_RESTART_THRESHOLD", 5)
                        )
                        logger.debug(
                            "User override value for slurm job restart: %i",
                            user_slurm_restart_thresh,
                        )
                        self.check_and_change_partition(
                            job_details=jdict,
                            restart_threshold=user_slurm_restart_thresh,
                        )
                if signal > 0:
                    # A non-zero signal may indicate e.g. an out-of-memory killed job
                    status = 128 + signal
                logger.debug(
                    "%s exit code of job %d is %s, return status %d",
                    args[0],
                    job_id,
                    exitcode,
                    status,
                )
                job_statuses[job_id] = state, status, reason
            logger.debug("%s returning job statuses: %s", args[0], job_statuses)
            return job_statuses

        def _getJobDetailsFromScontrol(self, job_id_list: list) -> dict:
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
            if isinstance(stdout, str):
                job_records = stdout.strip().split("\n\n")
            elif isinstance(stdout, bytes):
                job_records = stdout.decode("utf-8").strip().split("\n\n")

            # Collect the job statuses in a dict; key is the job-id, value is a tuple containing
            # job state and exit status. Initialize dict before processing output of `scontrol`.
            job_statuses = {}
            for job_id in job_id_list:
                job_statuses[job_id] = (None, None, None)

            # `scontrol` will report "No jobs in the system", if there are no jobs in the system,
            # and if no job-id was passed as argument to `scontrol`.
            if len(job_records) > 0 and job_records[0] == "No jobs in the system":
                return job_statuses

            for record in job_records:
                job = {}
                for line in record.splitlines():
                    for item in line.split():
                        # Output is in the form of many key=value pairs, multiple pairs on each line
                        # and multiple lines in the output. Each pair is pulled out of each line and
                        # added to a dictionary.
                        # Note: In some cases, the value itself may contain white-space. So, if we find
                        # a key without a value, we consider that key part of the previous value.
                        bits = item.split("=", 1)
                        if len(bits) == 1:
                            job[key] += " " + bits[0]
                        else:
                            key = bits[0]
                            job[key] = bits[1]
                    # The first line of the record contains the JobId. Stop processing the remainder
                    # of this record, if we're not interested in this job.
                    job_id = int(job["JobId"])
                    if job_id not in job_id_list:
                        logger.debug("%s job %d is not in the list", args[0], job_id)
                        break
                if job_id not in job_id_list:
                    continue
                state = job["JobState"]
                state = self._canonicalize_state(state)
                reason = ""
                if state == "PENDING":
                    reason = job.get("Reason")
                    if reason == "BadConstraints":
                        job["ExitCode"] = 7
                    if reason == "BeginTime":
                        logger.debug(
                            "Job: %s is in %s state. Checking if alternate partition to be used.",
                            job_id,
                            state,
                        )
                        user_slurm_restart_thresh = int(
                            os.getenv("TOIL_SLURM_JOB_RESTART_THRESHOLD", 5)
                        )
                        logger.debug(
                            "User override value for slurm job restart: %i",
                            user_slurm_restart_thresh,
                        )
                        self.check_and_change_partition(
                            job_details=job, restart_threshold=user_slurm_restart_thresh
                        )
                logger.debug("%s state of job %s is %s", args[0], job_id, state)
                try:
                    exitcode = job["ExitCode"]
                    if exitcode is not None:
                        status, signal = (int(n) for n in exitcode.split(":"))
                        if signal > 0:
                            # A non-zero signal may indicate e.g. an out-of-memory killed job
                            status = 128 + signal
                        logger.debug(
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
            logger.debug("%s returning job statuses: %s", args[0], job_statuses)
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
            job_environment: Optional[Dict[str, str]],
            gpus: Optional[int],
            usePreferredPartition: Optional[bool],
            comment: Optional[str],
        ) -> List[str]:
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
            nativeConfig = os.getenv("TOIL_SLURM_ARGS")

            # --export=[ALL,]<environment_toil_variables>
            set_exports = "--export=ALL"

            if nativeConfig is not None:
                logger.debug(
                    "Native SLURM options appended to sbatch from TOIL_SLURM_ARGS env. variable: %s",
                    nativeConfig,
                )

                for arg in nativeConfig.split():
                    if arg.startswith("--mem") or arg.startswith("--cpus-per-task"):
                        raise ValueError(
                            f"Some resource arguments are incompatible: {nativeConfig}"
                        )
                    # repleace default behaviour by the one stated at TOIL_SLURM_ARGS
                    if arg.startswith("--export"):
                        set_exports = arg
                sbatch_line.extend(nativeConfig.split())

            if environment:
                argList = []

                for k, v in environment.items():
                    quoted_value = quote(os.environ[k] if v is None else v)
                    argList.append(f"{k}={quoted_value}")

                set_exports += "," + ",".join(argList)

            # add --export to the sbatch
            sbatch_line.append(set_exports)

            parallel_env = os.getenv("TOIL_SLURM_PE")
            if cpu and cpu > 1 and parallel_env:
                sbatch_line.append(f"--partition={parallel_env}")

            if mem is not None and self.boss.config.allocate_mem:
                # memory passed in is in bytes, but slurm expects megabytes
                slurm_mem = math.ceil(mem / 2**20)
                sbatch_line.append(f"--mem={slurm_mem}")
            else:
                slurm_mem = None
            if cpu is not None:
                slurm_cpu = math.ceil(cpu)
                sbatch_line.append(f"--cpus-per-task={slurm_cpu}")

            if slurm_mem and slurm_cpu:
                # logger.info(
                #     "Trying to select partition based on cpus: %s and memory: %s of preferred type: %s",
                #     slurm_cpu,
                #     slurm_mem,
                #     usePreferredPartition,
                # )
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

        def parse_elapsed(self, elapsed):
            # slurm returns elapsed time in days-hours:minutes:seconds format
            # Sometimes it will only return minutes:seconds, so days may be omitted
            # For ease of calculating, we'll make sure all the delimeters are ':'
            # Then reverse the list so that we're always counting up from seconds -> minutes -> hours -> days
            total_seconds = 0
            try:
                elapsed = elapsed.replace("-", ":").split(":")
                elapsed.reverse()
                seconds_per_unit = [1, 60, 3600, 86400]
                for index, multiplier in enumerate(seconds_per_unit):
                    if index < len(elapsed):
                        total_seconds += multiplier * int(elapsed[index])
            except ValueError:
                pass  # slurm may return INVALID instead of a time
            return total_seconds

    # def _check_accelerator_request(self, requirer: Requirer) -> None:
    #     for accelerator in requirer.accelerators:
    #         if accelerator['kind'] != 'gpu':
    #             raise InsufficientSystemResources(requirer, 'accelerators', details=
    #                 [
    #                     f'The accelerator {accelerator} could not be provided'
    #                     'The Toil Slurm batch system only supports gpu accelerators at the moment.'
    #                 ])

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

        config_data = pd.DataFrame.from_dict(config_dicts)
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
    def add_options(cls, parser: Union[ArgumentParser, _ArgumentGroup]):
        allocate_mem = parser.add_mutually_exclusive_group()
        allocate_mem_help = (
            "A flag that can block allocating memory with '--mem' for job submissions "
            "on SLURM since some system servers may reject any job request that "
            "explicitly specifies the memory allocation.  The default is to always allocate memory."
        )
        allocate_mem.add_argument(
            "--dont_allocate_mem",
            action="store_false",
            dest="allocate_mem",
            help=allocate_mem_help,
        )
        allocate_mem.add_argument(
            "--allocate_mem",
            action="store_true",
            dest="allocate_mem",
            help=allocate_mem_help,
        )
        allocate_mem.set_defaults(allocate_mem=True)

    OptionType = TypeVar("OptionType")

    @classmethod
    def setOptions(cls, setOption: OptionSetter) -> None:
        setOption("allocate_mem")
