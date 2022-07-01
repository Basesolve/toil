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
from pipes import quote
from typing import Callable, Dict, List, Optional, TypeVar, Union
import configparser
import pandas as pd
from toil.batchSystems.abstractGridEngineBatchSystem import (
    AbstractGridEngineBatchSystem,
)
from toil.lib.misc import CalledProcessErrorStderr, call_command

logger = logging.getLogger(__name__)


class SlurmBatchSystem(AbstractGridEngineBatchSystem):
    class Worker(AbstractGridEngineBatchSystem.Worker):

        def getRunningJobIDs(self):
            # Should return a dictionary of Job IDs and number of seconds
            times = {}
            with self.runningJobsLock:
                currentjobs = {str(self.batchJobIDs[x][0]): x for x in self.runningJobs}
            # currentjobs is a dictionary that maps a slurm job id (string) to our own internal job id
            # squeue arguments:
            # -h for no header
            # --format to get jobid i, state %t and time days-hours:minutes:seconds

            lines = call_command(['squeue', '-h', '--format', '%i %t %M']).split('\n')
            for line in lines:
                values = line.split()
                if len(values) < 3:
                    continue
                slurm_jobid, state, elapsed_time = values
                if slurm_jobid in currentjobs and state == 'R':
                    seconds_running = self.parse_elapsed(elapsed_time)
                    times[currentjobs[slurm_jobid]] = seconds_running

            return times

        def killJob(self, jobID):
            call_command(['scancel', self.getBatchSystemID(jobID)])

        def prepareSubmission(self,
                              cpu: int,
                              memory: int,
                              jobID: int,
                              command: str,
                              jobName: str,
                              job_environment: Optional[Dict[str, str]] = None,
                              use_preferred_partition: Optional[bool] = True,
                              comment: Optional[str] = None,) -> List[str]:
            return self.prepareSbatch(cpu, memory, jobID, jobName, job_environment, use_preferred_partition, comment) + [f'--wrap={command}']

        def submitJob(self, subLine):
            try:
                output = call_command(subLine)
                # sbatch prints a line like 'Submitted batch job 2954103'
                result = int(output.strip().split()[-1])
                logger.debug("sbatch submitted job %d", result)
                return result
            except OSError as e:
                logger.error("sbatch command failed")
                raise e

        def coalesce_job_exit_codes(self, batch_job_id_list: list) -> list:
            """
            Collect all job exit codes in a single call.
            :param batch_job_id_list: list of Job ID strings, where each string has the form
            "<job>[.<task>]".
            :return: list of job exit codes, associated with the list of job IDs.
            """
            logger.debug("Getting exit codes for slurm jobs: %s", batch_job_id_list)
            # Convert batch_job_id_list to list of integer job IDs.
            job_id_list = [int(id.split('.')[0]) for id in batch_job_id_list]
            status_dict = self._get_job_details(job_id_list)
            exit_codes = []
            for _, status in status_dict.items():
                exit_codes.append(self._get_job_return_code(status))
            return exit_codes

        def getJobExitCode(self, batchJobID: str) -> int:
            """
            Get job exit code for given batch job ID.
            :param batchJobID: string of the form "<job>[.<task>]".
            :return: integer job exit code.
            """
            logger.debug("Getting exit code for slurm job: %s", batchJobID)
            # Convert batchJobID to an integer job ID.
            job_id = int(batchJobID.split('.')[0])
            status_dict = self._get_job_details([job_id])
            status = status_dict[job_id]
            return self._get_job_return_code(status)

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

        def _get_job_return_code(self, status: tuple) -> list:
            """
            Helper function for `getJobExitCode` and `coalesce_job_exit_codes`.
            :param status: tuple containing the job's state and it's return code.
            :return: the job's return code if it's completed, otherwise None.
            """
            state, rc = status
            # If job is in a running state, set return code to None to indicate we don't have
            # an update.
            if state in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'RESIZING', 'SUSPENDED'):
                rc = None
            return rc
        
        def check_and_change_partition(self, job_id, partition, restart_threshold=1):
            '''Get the job restart count and switch partition.
            restart_threshold=-1 implies node count per partition.

            :param job_id: pending jobs id
            :type job_id: int
            :param restart_threshold: number of restarts to wait for before updating partition, defaults to 4
            :type restart_threshold: int, optional
            '''
            restart_count = int(os.popen(
                f"""
                scontrol -o show job {job_id} |
                sed 's/ /\\n/g' |
                grep Restarts |
                cut -d "=" -f2
                """
            ).read().strip())
            alternate_partition = os.popen(
                f"""
                scontrol -o show partition {partition} |
                sed 's/ /\\n/g' |
                egrep 'TotalNodes|Alternate' |
                cut -d "=" -f2
                """
            ).read().strip()
            # set max_possible restart threshold
            # if restart_threshold == -1:
            #     restart_threshold = total_nodes
            if restart_count > restart_threshold and alternate_partition:
                logger.info(
                    "Job %s seems to have restarted by slurm beyond the threshold %s. Switching to alternate partition %s",
                    job_id, restart_threshold, alternate_partition
                )
                switch_exit_code = os.system(
                    f"""scontrol update jobid={job_id} partition={alternate_partition}"""
                )
                if switch_exit_code == 0:
                    logger.info(f"Job: %s has been swithced to alternate partition: %s", job_id, alternate_partition)
                else:
                    logger.warning(f"Job: %s could not be swithced to alternate partition: %s", job_id, alternate_partition)
            else:
                logger.info("Cannot switch partition: slurm job restart threshold exceeded but no alternate partition configured for %s", partition)

        def _getJobDetailsFromSacct(self, job_id_list: list) -> dict:
            """
            Get SLURM job exit codes for the jobs in `job_id_list` by running `sacct`.
            :param job_id_list: list of integer batch job IDs.
            :return: dict of job statuses, where key is the job-id, and value is a tuple
            containing the job's state and exit code.
            """
            job_ids = ",".join(str(id) for id in job_id_list)
            args = ['sacct',
                    '-n',  # no header
                    '-j', job_ids,  # job
                    '--format', 'JobIDRaw,State,ExitCode,Partition',  # specify output columns
                    '-P',  # separate columns with pipes
                    '-S', '1970-01-01']  # override start time limit
            stdout = call_command(args)

            # Collect the job statuses in a dict; key is the job-id, value is a tuple containing
            # job state and exit status. Initialize dict before processing output of `sacct`.
            job_statuses = {}
            for job_id in job_id_list:
                job_statuses[job_id] = (None, None)

            for line in stdout.splitlines():
                #logger.debug("%s output %s", args[0], line)
                values = line.strip().split('|')
                if len(values) < 3:
                    continue
                job_id_raw, state, exitcode, partition = values
                logger.debug("%s state of job %s is %s", args[0], job_id_raw, state)
                # JobIDRaw is in the form JobID[.JobStep]; we're not interested in job steps.
                job_id_parts = job_id_raw.split(".")
                if len(job_id_parts) > 1:
                    continue
                job_id = int(job_id_parts[0])
                status, signal = [int(n) for n in exitcode.split(':')]
                if state == 'PENDING':
                    logger.debug("Job: %s is in %s state. Checking if alternate partition to be used.", job_id, status)
                    self.check_and_change_partition(job_id=job_id, partition=partition)
                if signal > 0:
                    # A non-zero signal may indicate e.g. an out-of-memory killed job
                    status = 128 + signal
                logger.debug("%s exit code of job %d is %s, return status %d",
                             args[0], job_id, exitcode, status)
                job_statuses[job_id] = state, status
            logger.debug("%s returning job statuses: %s", args[0], job_statuses)
            return job_statuses

        def _getJobDetailsFromScontrol(self, job_id_list: list) -> dict:
            """
            Get SLURM job exit codes for the jobs in `job_id_list` by running `scontrol`.
            :param job_id_list: list of integer batch job IDs.
            :return: dict of job statuses, where key is the job-id, and value is a tuple
            containing the job's state and exit code.
            """
            args = ['scontrol',
                    'show',
                    'job']
            # `scontrol` can only return information about a single job,
            # or all the jobs it knows about.
            if len(job_id_list) == 1:
                args.append(str(job_id_list[0]))

            stdout = call_command(args)

            # Job records are separated by a blank line.
            if isinstance(stdout, str):
                job_records = stdout.strip().split('\n\n')
            elif isinstance(stdout, bytes):
                job_records = stdout.decode('utf-8').strip().split('\n\n')

            # Collect the job statuses in a dict; key is the job-id, value is a tuple containing
            # job state and exit status. Initialize dict before processing output of `scontrol`.
            job_statuses = {}
            for job_id in job_id_list:
                job_statuses[job_id] = (None, None)

            # `scontrol` will report "No jobs in the system", if there are no jobs in the system,
            # and if no job-id was passed as argument to `scontrol`.
            if len(job_records) > 0 and job_records[0] == "No jobs in the system":
                return job_statuses

            for record in job_records:
                job = {}
                for line in record.splitlines():
                    for item in line.split():
                        #logger.debug("%s output %s", args[0], item)
                        # Output is in the form of many key=value pairs, multiple pairs on each line
                        # and multiple lines in the output. Each pair is pulled out of each line and
                        # added to a dictionary.
                        # Note: In some cases, the value itself may contain white-space. So, if we find
                        # a key without a value, we consider that key part of the previous value.
                        bits = item.split('=', 1)
                        if len(bits) == 1:
                            job[key] += ' ' + bits[0]
                        else:
                            key = bits[0]
                            job[key] = bits[1]
                    # The first line of the record contains the JobId. Stop processing the remainder
                    # of this record, if we're not interested in this job.
                    job_id = int(job['JobId'])
                    if job_id not in job_id_list:
                        logger.debug("%s job %d is not in the list", args[0], job_id)
                        break
                if job_id not in job_id_list:
                    continue
                state = job['JobState']
                partition = job['Partition']
                if state == 'PENDING':
                    logger.info("Job: %s is in %s state. Checking if alternate partition to be used.", job_id, status)
                    self.check_and_change_partition(job_id=job_id, partition=partition)
                logger.debug("%s state of job %s is %s", args[0], job_id, state)
                try:
                    exitcode = job['ExitCode']
                    if exitcode is not None:
                        status, signal = [int(n) for n in exitcode.split(':')]
                        if signal > 0:
                            # A non-zero signal may indicate e.g. an out-of-memory killed job
                            status = 128 + signal
                        logger.debug("%s exit code of job %d is %s, return status %d",
                                     args[0], job_id, exitcode, status)
                        rc = status
                    else:
                        rc = None
                except KeyError:
                    rc = None
                job_statuses[job_id] = (state, rc)
            logger.debug("%s returning job statuses: %s", args[0], job_statuses)
            return job_statuses

        ###
        ### Implementation-specific helper methods
        ###
        def select_partition(self, cpus, mem, preferred=True):
            '''Select suitable slurm partition based on requirements. Checks state of partition.

            :param cpus: required cps
            :type cpus: int
            :param mem: required memory
            :type mem: integer
            :param preferred: respect partition preference, defaults to True
            :type preferred: bool, optional
            :return: suitable partition for the requirements
            :rtype: str
            '''
            if 'preference' in self.batchSystemResources.columns:
                possible_partitions = self.batchSystemResources.partitions[
                    (self.batchSystemResources['cputot'] >= cpus) &
                    (self.batchSystemResources['realmemory'] >= mem) &
                    (self.batchSystemResources['preference'] == preferred)
                ].values
            else:
                possible_partitions = self.batchSystemResources.partitions[
                    (self.batchSystemResources['cputot'] >= cpus) &
                    (self.batchSystemResources['realmemory'] >= mem)
                ].values
            if len(possible_partitions) != 0:
                usable_partitions = []
                logger.info("Available Partitions: %s", possible_partitions)
                for partition in possible_partitions:
                    partition_state = os.popen(
                        f"""
                        scontrol -o show partition {partition} |
                        sed 's/ /\\n/g' |
                        grep State |
                        cut -d "=" -f2
                        """
                    ).read().strip()
                    if partition_state == 'UP':
                        usable_partitions.append(partition)
                    else:
                        logger.info(
                            "Skipping partition: %s, due to state being %s",
                            partition,
                            partition_state
                        )
                logger.info("Usable Partitions: %s", usable_partitions)
                return usable_partitions[0]
            
            if 'preference' in self.batchSystemResources.columns:
                logger.warning(
                    "Could not find a partition to suffice cpus: %s, memory: %s and preferred type: %s",
                    cpus,
                    mem,
                    preferred
                )
                logger.info("Trying with preferred type: %s", not preferred)
                return self.select_partition(cpus, mem, not preferred)
            else:
                logger.error(
                    "Could not find a partition to suffice cpus: %s, memory: %s",
                    cpus,
                    mem
                )
                return

        def prepareSbatch(self,
                          cpu: int,
                          mem: int,
                          jobID: int,
                          jobName: str,
                          job_environment: Optional[Dict[str, str]],
                          use_preferred_partition: Optional[bool],
                          comment: Optional[str]) -> List[str]:

            #  Returns the sbatch command line before the script to run
            sbatch_line = ['sbatch', '-J', f'toil_job_{jobID}_{jobName}']

            environment = {}
            environment.update(self.boss.environment)
            if job_environment:
                environment.update(job_environment)

            if environment:
                argList = []

                for k, v in environment.items():
                    quoted_value = quote(os.environ[k] if v is None else v)
                    argList.append(f'{k}={quoted_value}')

                sbatch_line.append('--export=' + ','.join(argList))
            if mem is not None and self.boss.config.allocate_mem:
                # memory passed in is in bytes, but slurm expects megabytes
                slurm_mem = math.ceil(mem / 2 ** 20)
                sbatch_line.append(f'--mem={slurm_mem}')
            else:
                slurm_mem = None
            if cpu is not None:
                slurm_cpu = math.ceil(cpu)
                sbatch_line.append(f'--cpus-per-task={slurm_cpu}')

            if slurm_mem and slurm_cpu:
                logger.info(
                    "Trying to select partition based on cpus: %s and memory: %s of preferred type: %s",
                    slurm_cpu,
                    slurm_mem,
                    use_preferred_partition
                )
                partition = self.select_partition(
                    slurm_cpu,
                    slurm_mem,
                    preferred=use_preferred_partition
                )
                logger.info(
                    "Selected partition: %s based on cpus: %s and memory: %s of preferred type: %s",
                    partition,
                    slurm_cpu,
                    slurm_mem,
                    use_preferred_partition
                )
                sbatch_line.append(f'--partition={partition}')
            else:
                logger.info("Skipping slurm partition selection as mem and cpu are not specified.")
            
            if comment is not None:
                sbatch_line.append(f'--comment={comment}')

            stdoutfile: str = self.boss.formatStdOutErrPath(jobID, '%j', 'out')
            stderrfile: str = self.boss.formatStdOutErrPath(jobID, '%j', 'err')
            sbatch_line.extend(['-o', stdoutfile, '-e', stderrfile])

            # "Native extensions" for SLURM (see DRMAA or SAGA)
            nativeConfig = os.getenv('TOIL_SLURM_ARGS')
            if nativeConfig is not None:
                logger.debug("Native SLURM options appended to sbatch from TOIL_SLURM_ARGS env. variable: %s", nativeConfig)
                if ("--mem" in nativeConfig) or ("--cpus-per-task" in nativeConfig):
                    raise ValueError(f"Some resource arguments are incompatible: {nativeConfig}")

                sbatch_line.extend(nativeConfig.split())

            return sbatch_line

        def parse_elapsed(self, elapsed):
            # slurm returns elapsed time in days-hours:minutes:seconds format
            # Sometimes it will only return minutes:seconds, so days may be omitted
            # For ease of calculating, we'll make sure all the delimeters are ':'
            # Then reverse the list so that we're always counting up from seconds -> minutes -> hours -> days
            total_seconds = 0
            try:
                elapsed = elapsed.replace('-', ':').split(':')
                elapsed.reverse()
                seconds_per_unit = [1, 60, 3600, 86400]
                for index, multiplier in enumerate(seconds_per_unit):
                    if index < len(elapsed):
                        total_seconds += multiplier * int(elapsed[index])
            except ValueError:
                pass  # slurm may return INVALID instead of a time
            return total_seconds

    ###
    ### The interface for SLURM
    ###
    @classmethod
    def assessBatchResources(cls):
        slurm_partition_configs = os.popen(
            r"""
            scontrol show node -o |
            sed 's/\[.*//g' |
            sed 's/NodeName=/\[/' |
            sed 's/ /\] /' |
            sed 's/ \+/\n/g' |
            egrep "=|\["
            """
        ).read().strip()
        # print(slurm_partition_configs)
        config = configparser.ConfigParser()
        config.read_string(slurm_partition_configs)
        config_dicts = []
        for section, val in config._sections.items():
            cdict = {'NodeName': section}
            for k, v in val.items():
                cdict[k] = v
            config_dicts.append(cdict)

        config_data = pd.DataFrame.from_dict(config_dicts)
        # print(config_data)
        req_configs = config_data[
            [
                'partitions', 'cputot', 'realmemory'
            ]
        ]
        slurm_resources = req_configs.groupby("partitions").max().reset_index()
        preference = os.getenv("TOIL_SLURM_PARTITON_PREFERED")
        if preference:
            logger.info(
                "Setting slurm partition preference: %s",
                preference
            )
            slurm_resources['preference'] = slurm_resources.partitions.str.contains(preference, case=False)
        else:
            logger.info(
                "No slurm partition preference set"
            )
        slurm_resources[['cputot', 'realmemory']] = slurm_resources[
            ['cputot', 'realmemory']
        ].astype(int)
        slurm_resources.sort_values(['cputot', 'realmemory'], inplace=True)
        return slurm_resources

    @classmethod
    def getWaitDuration(cls):
        # Extract the slurm batchsystem config for the appropriate value
        lines = call_command(['scontrol', 'show', 'config']).split('\n')
        time_value_list = []
        for line in lines:
            values = line.split()
            if len(values) > 0 and (values[0] == "SchedulerTimeSlice" or values[0] == "AcctGatherNodeFreq"):
                time_name = values[values.index('=')+1:][1]
                time_value = int(values[values.index('=')+1:][0])
                if time_name == 'min':
                    time_value *= 60
                # Add a 20% ceiling on the wait duration relative to the scheduler update duration
                time_value_list.append(math.ceil(time_value*1.2))
        return max(time_value_list)

    @classmethod
    def add_options(cls, parser: Union[ArgumentParser, _ArgumentGroup]):
        allocate_mem = parser.add_mutually_exclusive_group()
        allocate_mem_help = ("A flag that can block allocating memory with '--mem' for job submissions "
                             "on SLURM since some system servers may reject any job request that "
                             "explicitly specifies the memory allocation.  The default is to always allocate memory.")
        allocate_mem.add_argument("--dont_allocate_mem", action='store_false', dest="allocate_mem", help=allocate_mem_help)
        allocate_mem.add_argument("--allocate_mem", action='store_true', dest="allocate_mem", help=allocate_mem_help)
        allocate_mem.set_defaults(allocate_mem=True)

    OptionType = TypeVar('OptionType')
    @classmethod
    def setOptions(cls, setOption: Callable[[str, Optional[Callable[[str], OptionType]], Optional[Callable[[OptionType], None]], Optional[OptionType]], None]) -> None:
        setOption("allocate_mem", bool, default=False)

