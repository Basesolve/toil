# Graph Report - toil  (2026-09-28)

## Corpus Check
- 360 files · ~535,418 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 8136 nodes · 20426 edges · 332 communities (283 shown, 49 thin omitted)
- Extraction: 82% EXTRACTED · 18% INFERRED · 0% AMBIGUOUS · INFERRED: 3733 edges (avg confidence: 0.52)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ee4afd0e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Job
- JobDescription
- InsufficientSystemResources
- AbstractFileStore
- toil/common.py
- MockBatchSystemAndProvisioner
- WDLBindings
- AbstractProvisioner
- Test
- Leader
- toil/test/__init__.py
- Shape
- SlurmTest
- AWSProvisioner
- AWSJobStore
- cwlTest.py
- GoogleJobStoreTest
- get_data
- ._read
- BatchJobExitReason
- GoogleJobStore
- HistoryManager
- gceProvisionerTest.py
- integrative
- statsAndLogging.py
- iam.py
- CWLObjectType
- .wrapJobFn
- main
- GridEngineThread
- MesosExecutor
- ToilMetrics
- trs.py
- Path
- slow
- .__init__
- fileStoreTest.py
- retry
- AbstractCachingFileStoreTest
- awsProvisioner.py
- AbstractStateStore
- .get_is_directory
- AWSBadEncryptionKeyError
- ._virtualize_filename
- GridEngineThread
- FileID
- e
- promisedRequirementTest.py
- log_bindings
- Base
- deferredFunctionTest.py
- cleanup_aws_resources.py
- TestUtils
- Any
- WDLKubernetesClusterTest
- S3Test
- .__init__
- serverTest.py
- GCEProvisioner
- toil/lib/aws/__init__.py
- TestJobService
- Appliance
- MesosBatchSystem
- needs_docker
- tasks.py
- Any
- .testConcurrencyWithDisk
- toilStats.py
- wes_cwl_runner.py
- abstractBatchSystem.py
- .prepareSbatch
- global_mutex
- Node
- ToilBackend
- AbstractToilWESServerTest
- decode_directory
- trim_mounts_op_up
- TestSlurmMountRecovery
- .report_on_jobs
- checkDockerImageExists
- dockstore.py
- ToilDocumentationTest
- .get_dependencies
- lib.py
- .getLogger
- GridEngineThread
- RuntimeError
- call_command
- LastProcessStandingArena
- FileJobStore
- test_ec2.py
- ensure_file_imported
- ._parseResource
- version_template.py
- ._connect
- toil/__init__.py
- .session
- threading.py
- Submission
- .add_toil_service
- lsfHelper.py
- Config
- AbstractJobStore
- ValueError
- makeRootJob
- JobStatus
- ToilRestartException
- leader.py
- apiDockerCall
- .restartCheckpoint
- AtomicFileCreate
- WESBackend
- history_submission.py
- options/common.py
- GridEngineThread
- slurm_mount_recovery.py
- generate_default_job_store
- SortTest
- ._get_blob_from_url
- TestJob
- skip
- worker.py
- toilDebugTest.py
- url.py
- slurmSortTest.py
- InstanceConfiguration
- WorkflowStateStore
- WorkflowStateMachine
- UserDefinedJobArgTypeTest
- plugins.py
- Info
- toil_backend.py
- ToilWorkflow
- .__init__
- sortTest.py
- CachingFileStore
- stress.py
- ec2nodes.py
- UserNameUnvailableTest
- ServiceManager
- jobServiceTest.py
- TestImportExportFile
- toil-sort-example.py
- .read_from_url
- ._file_maker
- setter
- toilContextManagerTest.py
- directory.py
- RealtimeLogger
- DirectoryResource
- MemoryStateCache
- .test_symlink_read_control
- AutoDeploymentTest
- TestCleanWorkDir
- TestWDLToilBench
- create_current_submission
- slurm.py
- TestSafeFileInterleaving
- ToilTest
- utilsTest.py
- conf.py
- ._createJobStore
- awsBatch.py
- executor
- resumabilityTest.py
- Toil Job Store Storage — Cost & Architecture Analysis
- get_job_kind
- TestResource
- .testGetStatusFailedCWLWF
- strtobool
- .run
- Contributor Covenant Code of Conduct
- JuiceFS — performance, stability, and Toil fit
- .formatLogStream
- history.py
- ._loadUserModule
- .killJobs
- .handle_injection_messages
- environmentTest.py
- .test_cwl_toil_kill
- wdltoil_test.py
- Notes for AI Assistants
- .forModule
- Production job counts and S3 restart latency
- .testAWSProvisionerUtils
- uploadFile
- conversions.py
- replay_message_bus
- safeFileTest.py
- LockCheckpointer
- find_hanging_tests.py
- ReadableFileObj
- realtimeLogger.py
- find_default_container
- Side-by-side monthly totals
- Cost models compared
- TaskRunner
- Recommendations
- url_plugin_test.py
- restartDAGTest.py
- .addService
- .addChild
- .__getstate__
- MessageInbox
- addOptions
- .url_exists
- ._setLeaderWorkerAuthentication
- scripts/tutorial_stats.py
- google_retry
- connect_to_workflow_state_store
- checksum.py
- slurm_recovery_workflow.py
- examples/example_cachingbenchmark.py
- .open
- ._runOrphanedDeferredFunctions
- ROADMAP.md
- get_object_for_url
- .wrapFn
- helloWorldTest.py
- fn1Test
- realtimeLoggerTest.py
- helloWorld.py
- .run
- scripts/example_cachingbenchmark.py
- .fetch_url
- Checkpoint
- AWS storage pricing reference (approximate)
- add_paths
- ._try_terminate
- Current Toil AWS job store behavior
- .scan_bus_messages
- Non-cost tradeoffs
- HelloWorld
- HelloWorld
- PULL_REQUEST_TEMPLATE.md
- .with_retries
- .get_alternate_partition
- .cachingIsFree
- IAMTest
- .test_writer_paused_mid_write_blocks_reader
- .setupJobAfterFailure
- get_requirements
- .getDefaultOptions
- ._read_contents
- .__init__
- ImportWorkersMessageHandler
- Runner
- .getNodeShape
- safe_write_file
- .testStatsAndLogging
- AGENTS.md
- strip_trailing_whitespace_from_all_files_in_dir
- .run
- TimeWaster
- toil/conftest.py
- .getDefaultArgumentParser
- ._get_job_store_classes
- .testPartialReadFromStream
- .get_toil_coordination_dir
- .testFileDeletion
- FileResource
- .private_history_manager
- test-pr
- wheel-of-issues
- .getCacheExtraJobSpace
- .testImportHttpFile
- .readGlobalFileStream
- applianceSelf
- .testZeroLengthFiles
- enable_absolute_imports
- Dockerfile.py
- wdltoil.py
- ._testJobFileStore
- ._corruptJobStore
- rootpath
- .add_options
- .testEmptyFileStoreIDIsReadable
- .getLocalTempFile
- .testInitialState
- .testJobCreation
- .testChildLoadingEquality
- .testUpdateBehavior
- .testJobDeletions
- .testReadWriteSharedFilesTextMode
- .testReadWriteFileStreamTextMode
- .testPerJobFiles
- .testBatchCreate
- .testGrowingAndShrinkingJob
- ._prepareTestFile
- .getWorkerContexts
- .add_options
- .con
- atomic_copyobj
- .__init__
- podKiller.sh
- check_out.sh script
- run.sh
- customDockerInit.sh
- singularity-wrapper.sh
- waitForKey.sh
- sphinxcontrib/__init__.py

## God Nodes (most connected - your core abstractions)
1. `Job` - 389 edges
2. `Toil` - 363 edges
3. `JobDescription` - 260 edges
4. `Config` - 259 edges
5. `AbstractJobStore` - 253 edges
6. `AbstractFileStore` - 208 edges
7. `FileJobStore` - 155 edges
8. `slow()` - 155 edges
9. `URLAccess` - 137 edges
10. `InsufficientSystemResources` - 114 edges

## Surprising Connections (you probably didn't know these)
- `find_buckets_to_cleanup()` --indirect_call--> `e()`  [INFERRED]
  contrib/admin/cleanup_aws_resources.py → src/toil/test/src/promisesTest.py
- `find_sdb_domains_to_cleanup()` --indirect_call--> `e()`  [INFERRED]
  contrib/admin/cleanup_aws_resources.py → src/toil/test/src/promisesTest.py
- `in_acceptable_environment()` --calls--> `inVirtualEnv()`  [INFERRED]
  contrib/hooks/lib.py → src/toil/__init__.py
- `parser_with_common_options()` --indirect_call--> `version()`  [INFERRED]
  src/toil/common.py → version_template.py
- `main()` --calls--> `Toil`  [INFERRED]
  examples/example_alwaysfail.py → src/toil/common.py

## Import Cycles
- 3-file cycle: `src/toil/batchSystems/abstractBatchSystem.py -> src/toil/batchSystems/options.py -> src/toil/batchSystems/registry.py -> src/toil/batchSystems/abstractBatchSystem.py`
- 3-file cycle: `src/toil/batchSystems/abstractBatchSystem.py -> src/toil/common.py -> src/toil/realtimeLogger.py -> src/toil/batchSystems/abstractBatchSystem.py`
- 3-file cycle: `src/toil/batchSystems/abstractBatchSystem.py -> src/toil/deferred.py -> src/toil/realtimeLogger.py -> src/toil/batchSystems/abstractBatchSystem.py`
- 3-file cycle: `src/toil/provisioners/__init__.py -> src/toil/provisioners/gceProvisioner.py -> src/toil/provisioners/abstractProvisioner.py -> src/toil/provisioners/__init__.py`
- 3-file cycle: `src/toil/provisioners/__init__.py -> src/toil/provisioners/aws/awsProvisioner.py -> src/toil/provisioners/abstractProvisioner.py -> src/toil/provisioners/__init__.py`
- 3-file cycle: `src/toil/lib/io.py -> src/toil/lib/memoize.py -> src/toil/lib/threading.py -> src/toil/lib/io.py`
- 4-file cycle: `src/toil/batchSystems/abstractBatchSystem.py -> src/toil/common.py -> src/toil/batchSystems/options.py -> src/toil/batchSystems/registry.py -> src/toil/batchSystems/abstractBatchSystem.py`
- 4-file cycle: `src/toil/batchSystems/abstractBatchSystem.py -> src/toil/fileStores/abstractFileStore.py -> src/toil/common.py -> src/toil/realtimeLogger.py -> src/toil/batchSystems/abstractBatchSystem.py`
- 4-file cycle: `src/toil/batchSystems/abstractBatchSystem.py -> src/toil/job.py -> src/toil/deferred.py -> src/toil/realtimeLogger.py -> src/toil/batchSystems/abstractBatchSystem.py`
- 5-file cycle: `src/toil/batchSystems/abstractBatchSystem.py -> src/toil/common.py -> src/toil/options/common.py -> src/toil/batchSystems/options.py -> src/toil/batchSystems/registry.py -> src/toil/batchSystems/abstractBatchSystem.py`
- 5-file cycle: `src/toil/batchSystems/abstractBatchSystem.py -> src/toil/fileStores/abstractFileStore.py -> src/toil/common.py -> src/toil/batchSystems/options.py -> src/toil/batchSystems/registry.py -> src/toil/batchSystems/abstractBatchSystem.py`
- 5-file cycle: `src/toil/batchSystems/abstractBatchSystem.py -> src/toil/common.py -> src/toil/job.py -> src/toil/deferred.py -> src/toil/realtimeLogger.py -> src/toil/batchSystems/abstractBatchSystem.py`
- 5-file cycle: `src/toil/batchSystems/abstractBatchSystem.py -> src/toil/fileStores/abstractFileStore.py -> src/toil/fileStores/cachingFileStore.py -> src/toil/common.py -> src/toil/realtimeLogger.py -> src/toil/batchSystems/abstractBatchSystem.py`
- 5-file cycle: `src/toil/batchSystems/abstractBatchSystem.py -> src/toil/fileStores/abstractFileStore.py -> src/toil/fileStores/nonCachingFileStore.py -> src/toil/common.py -> src/toil/realtimeLogger.py -> src/toil/batchSystems/abstractBatchSystem.py`
- 5-file cycle: `src/toil/batchSystems/abstractBatchSystem.py -> src/toil/fileStores/abstractFileStore.py -> src/toil/job.py -> src/toil/deferred.py -> src/toil/realtimeLogger.py -> src/toil/batchSystems/abstractBatchSystem.py`
- 5-file cycle: `src/toil/lib/io.py -> src/toil/lib/url.py -> src/toil/lib/plugins.py -> src/toil/lib/memoize.py -> src/toil/lib/threading.py -> src/toil/lib/io.py`

## Communities (332 total, 49 thin omitted)

### Community 0 - "Job"
Cohesion: 0.01
Nodes (203): LocalFileStoreJob, Names, Stores all the kinds of name a job can have., getFileSystemSize(), A context manager that represents a Toil workflow. Specifically the batch…, Return the free space, and total size of the file system hosting `dirPath`.…, Toil, Attach the Toil tool to the Toil job that is executing it. This allows it to… (+195 more)

### Community 1 - "JobDescription"
Cohesion: 0.01
Nodes (160): Neutral place for exceptions, to break import cycles., CheckpointJobDescription, JobDescription, JobException, Get the names and ID of this job as a named tuple., Get all the jobs that executed in this job's chain, in order. For each job,…, Find all batches of service host job IDs that can be started at the same time.…, Get an iterator over all child/follow-on/chained inherited successor job IDs,… (+152 more)

### Community 2 - "InsufficientSystemResources"
Cohesion: 0.03
Nodes (93): pneeds_mesos, AbstractBatchSystem, BatchSystemSupport, InsufficientSystemResources, ParsedRequirement, An abstract base class to represent the interface the batch system must provide…, Whether this batch system supports auto-deployment of the user script itself.…, Whether this batch system supports worker cleanup. Indicates whether this batch… (+85 more)

### Community 3 - "AbstractFileStore"
Cohesion: 0.03
Nodes (207): _Hash, PathMapper, InconsistentConfigurationError, Represents when a Toil configuration is nonsensical and cannot be used. Has an…, Conditional, CWLGather, CWLImportWrapper, CWLInstallImportsJob (+199 more)

### Community 4 - "toil/common.py"
Cohesion: 0.04
Nodes (17): binaryStrings(), merge(), DemoService, # TODO: Get ahold of the timing from statsAndLogging instead of redoing it here!, # TODO: I thought we refactored the different job store import, # TODO: If check_existence is false but a shared file name is, # TODO: why/how would this propagate when not using single machine?, # TODO: May interfere with workflow directory creation logging if it's (+9 more)

### Community 5 - "MockBatchSystemAndProvisioner"
Cohesion: 0.08
Nodes (4): MockBatchSystemAndProvisioner, Add a job to the job queue, Mimics a leader, job batcher, provisioner and scalable batch system., Returns a list of Node objects, each representing a worker node in the cluster…

### Community 6 - "WDLBindings"
Cohesion: 0.06
Nodes (53): Gather, Loader, Promised, T, Function for ensuring you actually have a promised value, and not just a…, Function for ensuring you actually have a collection of promised values, and…, unwrap(), unwrap_all() (+45 more)

### Community 7 - "AbstractProvisioner"
Cohesion: 0.07
Nodes (18): AbstractProvisioner, Interface for provisioning worker nodes to use in a Toil cluster., Initialize provisioner. Implementations should raise…, Get all the cluster types that this provisioner implementation supports., Initialize class for a new cluster, to be deployed, when running outside the…, Initialize class from an existing cluster. This method assumes that the…, Return the contents of the file written by `self._write_file_to_cloud()`., Forget any authentication information populated by… (+10 more)

### Community 8 - "Test"
Cohesion: 0.08
Nodes (11): memoize, Test importing a file over FTP, :rtype: AbstractJobStore, A test of job stores that use encryption, Create an encrypted file. Read it in encrypted mode then try with encryption…, Tests that a job created via one JobStore instance can be loaded from another., Make sure that updating a job persists filesToDelete. The following…, Checks if updating a stream and failing in the middle does nothing. (+3 more)

### Community 9 - "Leader"
Cohesion: 0.06
Nodes (31): JobUpdatedMessage, Produced when a job is "updated" and ready to have something happen to it., Leader, Check if the system is deadlocked running service jobs., Note that progress has been made and any pending deadlock checks should be…, Add a job to the queue of jobs currently trying to run., Add a list of jobs, each represented as a jobNode object., Issue a service job. Put it on a queue if the maximum number of service jobs to… (+23 more)

### Community 10 - "toil/test/__init__.py"
Cohesion: 0.06
Nodes (61): MT, Return the path with conforms to the given mode on the Path. [Copy-pasted in…, which(), have_working_nvidia_docker_runtime(), have_working_nvidia_smi(), Return True if the nvidia-smi binary, from nvidia's CUDA userspace utilities,…, Return True if Docker exists and can handle an "nvidia" runtime and the "--…, _is_kubernetes_installed_and_configured() (+53 more)

### Community 11 - "Shape"
Cohesion: 0.02
Nodes (87): FailedConstraint, AbstractScalableBatchSystem, NodeInfo, The coresUsed attribute is a floating point value between 0 (all cores idle)…, A batch system that supports a variable number of worker nodes. Used by…, Returns a dictionary mapping node identifiers of preemptible or non-preemptible…, Can be used to determine if a worker node is running any tasks. If the node is…, Stop sending jobs to this node. Used in autoscaling when the autoscaler is… (+79 more)

### Community 12 - "SlurmTest"
Cohesion: 0.04
Nodes (26): call_either(), call_sacct(), call_sacct_raises(), call_scontrol(), The arguments passed to `call_command` when executing `scontrol` are:…, Fake that the `sacct` command fails by raising a `CalledProcessErrorStderr`, Pretend to call either sacct or scontrol as appropriate., Class for unit-testing SlurmBatchSystem (+18 more)

### Community 13 - "AWSProvisioner"
Cohesion: 0.03
Nodes (69): Collection, E, EC2Client, FilterTypeDef, InstanceTypeDef, boto3_pager(), Any, Yield all the results from calling the given Boto 3 method with the given… (+61 more)

### Community 14 - "AWSJobStore"
Cohesion: 0.07
Nodes (27): AWSJobStore, Any, IO, Turn s3:// into http:// and put a public-read ACL on it., Turn s3:// into http:// and put a public-read ACL on it., Create an empty file in s3 and return a bare string file ID., This fetches all referenced logs in the database from s3 as readable objects…, Get the encryption arguments to pass to an AWS function. Reads live from the… (+19 more)

### Community 15 - "cwlTest.py"
Cohesion: 0.08
Nodes (58): cwl_small, cwl_small_log_dir, DownReturnType, LogCaptureFixture, Any, No-op function for mount-point trimming., Remove any redundant mount points from the listing. Modifies the CWL object in…, Apply the given operation to all top-level CWL objects with the given named CWL… (+50 more)

### Community 16 - "GoogleJobStoreTest"
Cohesion: 0.22
Nodes (3): google_retry(), GoogleJobStoreTest, xfail

### Community 17 - "get_data"
Cohesion: 0.06
Nodes (35): AbstractContextManager, Test that running a CWL workflow with inputs specified on the command line…, _fallback_get_data(), get_data(), needs_singularity_or_docker(), Path, Use as a decorator before test classes or methods to only run them if docker is…, Returns an absolute path for a file from this package. (+27 more)

### Community 18 - "._read"
Cohesion: 0.11
Nodes (18): If disk space is overcommitted, try one round of collecting files to…, If disk space is overcomitted, block and evict eligible things from the cache…, Creates a file in the jobstore and returns a FileID reference., Read a file without putting it into the cache. :param toil.fileStores.FileID…, Read a mutable copy of a file, putting it into the cache if possible. :param…, For use when you own a file in 'downloading' state, and have a 'copying'…, Move a downloaded file in 'downloading' state, owned by us, from the cache to a…, Read a file, putting it into the cache if possible. :param… (+10 more)

### Community 19 - "BatchJobExitReason"
Cohesion: 0.02
Nodes (112): BatchV1Api, CoreV1Api, CustomObjectsApi, OptionType, R, BatchJobExitReason, Set the user script for this workflow. This method must be called before the…, Returns information about job that has updated its status (i.e. ceased running,… (+104 more)

### Community 20 - "GoogleJobStore"
Cohesion: 0.14
Nodes (4): GoogleJobStore, Job store implementation backed by Google Cloud Storage., Return a dict of environment variables to send out to the workers so they can…, Yields a context manager that can be used to write to the bucket with a stream.…

### Community 21 - "HistoryManager"
Cohesion: 0.08
Nodes (39): db_retry(), HistoryManager, Connection, Cursor, RT, Mark a collection of job attempts as submitted to Dockstore in a single…, Count workflows in the database., Count workflow attempts in the database. (+31 more)

### Community 22 - "gceProvisionerTest.py"
Cohesion: 0.06
Nodes (26): needs_google_project(), needs_google_storage(), Use as a decorator before test classes or methods to run only if Google Cloud…, Use as a decorator before test classes or methods to run only if we have a…, AbstractGCEAutoscaleTest, GCEAutoscaleTest, GCEAutoscaleTestMultipleNodeTypes, GCERestartTest (+18 more)

### Community 23 - "integrative"
Cohesion: 0.03
Nodes (57): AWSRegionName, Get a region (e.g. us-west-2) from a zone (e.g. us-west-1c)., zone_to_region(), AWSConnectionManager, # TODO: mypy is going to have !!FUN!! with this API because the final type, Make a new empty AWSConnectionManager., Class that represents a connection to AWS. Caches Boto 3 and Boto 2 objects by…, Given an availability zone, get the default subnet for the default VPC in that… (+49 more)

### Community 24 - "statsAndLogging.py"
Cohesion: 0.05
Nodes (55): main(), generate_config(), parser_with_common_options(), ArgParser, Derive configuration from the command line options. Then load the job store…, Create an instance of the concrete job store implementation that matches the…, Write a Toil config file to the given path. Safe to run simultaneously in…, Get a command-line option parser for a Toil subcommand. The returned parser… (+47 more)

### Community 25 - "iam.py"
Cohesion: 0.13
Nodes (29): AllowedActionCollection, add_to_action_collection(), allowed_actions_attached(), allowed_actions_group(), allowed_actions_roles(), allowed_actions_user(), collect_policy_actions(), get_actions_from_policy_document() (+21 more)

### Community 26 - "CWLObjectType"
Cohesion: 0.10
Nodes (25): CWLObjectType, Process, RuntimeContext, determine_load_listing(), get_container_engine(), makeJob(), Logger, Promised (+17 more)

### Community 27 - ".wrapJobFn"
Cohesion: 0.04
Nodes (35): Makes a Job out of a job function. Convenience function for constructor of…, AbstractFileStoreTest, xfail, Write a couple of files to the jobstore. Delete a couple of them. Read back…, Write a couple of files to the jobstore. Delete a couple of them. Read back…, I do nothing. Don't judge me., Runs a simple DAG to test if if any features other that caching were broken., Write a couple of files to the jobstore. Delete a couple of them. Read back… (+27 more)

### Community 28 - "main"
Cohesion: 0.11
Nodes (18): CommentedMap, CWLTestConfig, LoadingContext, ProcessType, main(), TextIO, Emit custom ToilCommandLineTools. This factory function is meant to be passed…, Try to prepull all containers in a CWL workflow with Singularity or Docker.… (+10 more)

### Community 29 - "GridEngineThread"
Cohesion: 0.09
Nodes (16): GridEngineThread, JobTuple, Thread, Get batch system-specific job ID Note: for the moment this is the only…, Remove jobID passed :param jobID: toil job ID, Create a new job with the given attributes. Implementation-specific; called by…, Kill any running jobs within thread, Check and update status of all running jobs. Respects statePollingWait and will… (+8 more)

### Community 30 - "MesosExecutor"
Cohesion: 0.11
Nodes (10): Executor, MesosExecutor, Invoked when a fatal error has occurred with the executor and/or executor…, Invoked by SchedulerDriver when a Mesos task should be launched by this executor, Invoked when a framework message has arrived for this executor., Part of Toil's Mesos framework, runs on a Mesos agent. A Toil job is passed to…, Invoked once the executor driver has been able to successfully connect with…, Invoked when the executor re-registers with a restarted agent. (+2 more)

### Community 31 - "ToilMetrics"
Cohesion: 0.07
Nodes (31): ClusterDesiredSizeMessage, ClusterSizeMessage, gen_message_bus_path(), JobAnnotationMessage, JobCompletedMessage, JobFailedMessage, JobIssuedMessage, JobMissingMessage (+23 more)

### Community 32 - "trs.py"
Cohesion: 0.07
Nodes (28): compose_trs_spec(), extract_trs_spec(), fetch_workflow(), find_workflow(), is_trs_workflow(), parse_trs_spec(), Compose a TRS ID from a workflow ID and version ID., Given a Dockstore URL or TRS identifier, get the root WDL or CWL URL for the… (+20 more)

### Community 33 - "Path"
Cohesion: 0.08
Nodes (18): aws_s3, Path, Generate the putput we expect from load_contents.cwl, when sending output files…, CWL tests included in Toil that don't involve the whole CWL conformance test…, Helper function that runs a CWL workflow and checks the result. Implements…, Helper function that runs a CWL workflow with --debugWorker and checks the…, Run a generic download test with a tester function and check the result. Ther…, Test if CWL handles file: URIs without even empty hostnames. (+10 more)

### Community 34 - "slow"
Cohesion: 0.12
Nodes (13): Use this decorator to identify tests that are slow and not critical. Skip if…, slow(), docker, online, Path, r""" Test for piping API for dockerCall(). Using this API (activated when list…, By default, executing cmd1 | cmd2 | ... | cmdN, will only return an error if…, Test for the different log outputs when deatch=False. (+5 more)

### Community 35 - ".__init__"
Cohesion: 0.08
Nodes (21): Scatter, poll_execution_cache(), Any, Workflow, Make a WDL-related job. Makes sure the global recursive call limit is high…, Make a new job to determine resources and run a task. :param…, Make a new job to run a task. :param enclosing_bindings: Bindings outside the…, Make a new job to run a workflow node to completion. (+13 more)

### Community 36 - "fileStoreTest.py"
Cohesion: 0.13
Nodes (19): CacheUnbalancedError, IllegalDeletionCacheError, Raised if file store can't free enough space for caching, Error raised if the caching code discovers a file that represents a reference…, AbstractNonCachingFileStoreTest, CachingFileStoreTestWithAwsJobStore, CachingFileStoreTestWithFileJobStore, CachingFileStoreTestWithGoogleJobStore (+11 more)

### Community 37 - "retry"
Cohesion: 0.06
Nodes (56): Bucket, BucketLocationConstraintType, GetObjectOutputTypeDef, HeadObjectOutputTypeDef, ListMultipartUploadsOutputTypeDef, ObjectTypeDef, PutObjectOutputTypeDef, S3ServiceResource (+48 more)

### Community 38 - "AbstractCachingFileStoreTest"
Cohesion: 0.04
Nodes (30): Pack the FileID into a string so it can be passed through external code., AbstractCachingFileStoreTest, Write a non-local file to the job store(hence no cached copy), then have 10…, This function does what the two Multiple File reading tests want to do :param…, Read a file from the job store immutable and explicitly ask to have it in the…, Aux function for jobCacheTest.testReturnFileSizes Conduct numIters operations…, Assert the values for job disk and total cached file sizes tracked by the file…, This is the aux function for the controlled failed worker test. It does a… (+22 more)

### Community 39 - "awsProvisioner.py"
Cohesion: 0.06
Nodes (53): flatten_tags(), Convert tags from a key to value dict into a list of 'Key': xxx, 'Value': xxx…, create_auto_scaling_group(), create_instances(), create_launch_template(), create_ondemand_instances(), create_spot_instances(), inconsistencies_detected() (+45 more)

### Community 40 - "AbstractStateStore"
Cohesion: 0.10
Nodes (14): AbstractStateStore, FileStateStore, Safely read a file by acquiring a shared lock to prevent other processes from…, A place for the WES server to keep its state: the set of workflows that exist…, Get the value of the given key for the given workflow, or None if the key is…, Set the value of the given key for the given workflow. If the value is None,…, Read a value from a local cache, without checking the actual backend., Write a value to a local cache, without modifying the actual backend. (+6 more)

### Community 41 - ".get_is_directory"
Cohesion: 0.10
Nodes (9): Get the size of the object at the given URL, or None if it cannot be obtained., Return True if the thing at the given URL is a directory, and False if it is a…, List the contents of the given URL, which may or may not end in '/' Returns a…, Get the size in bytes of the file at the given URL, or None if it cannot be…, Return True if the thing at the given URL is a directory, and False if it is a…, List the directory at the given URL. Returned path components can be joined…, Test URLAccess class handling read, list, and checking the size/existence of…, TestURLAccess (+1 more)

### Community 42 - "AWSBadEncryptionKeyError"
Cohesion: 0.07
Nodes (34): Writes the contents of a file to a source (writes url to writable) using a…, AWSBadEncryptionKeyError, AWSKeyAlreadyExistsError, AWSKeyNotFoundError, download_stream(), Exception, Context manager that gives out a download stream to download data., ChecksumError (+26 more)

### Community 43 - "._virtualize_filename"
Cohesion: 0.12
Nodes (13): Apply, is_standard_url(), is_toil_url(), Return True if the given URL is a non-Toil, non-file: URL., Return True if a URL is a toilfile: or toildir: URL., Test to make sure Toil URI packing brings through the required information., pack_toil_uri(), Replacement evaluation implementation that avoids downloads. (+5 more)

### Community 44 - "GridEngineThread"
Cohesion: 0.09
Nodes (21): JobStatusDetail, GridEngineThread, env_float(), env_int(), Return True when opt-in spot/SIGTERM per-job failover is enabled., spot_failover_enabled(), datetime, Given a list of job IDs and a list of job details (state, exit code, reason),… (+13 more)

### Community 45 - "FileID"
Cohesion: 0.03
Nodes (32): Import the file at the given URL into the job store. By default, returns None…, Export file to destination pointed at by the destination URL. See…, Given a URI, if it has no scheme, make it a properly quoted file: URI. :param…, Upload a file (as a path) to the job store. If the file is in a FileStore-…, Similar to writeGlobalFile, but allows the writing of a stream to the job…, Record that the given file was read by the job. (to be announced if the job…, Get the size of the file pointed to by the given ID, in bytes. If a FileID or…, Delete local copies of files associated with the provided job store ID. Raises… (+24 more)

### Community 46 - "e"
Cohesion: 0.05
Nodes (58): copyKeyMultipart(), BaseException, RuntimeError, Raised when AWS refuses to perform a server-side copy between S3 keys, and…, Copies a key from a source key to a destination key in multiple parts. Note…, # TODO: This function is unused., retryable_ssl_error(), ServerSideCopyProhibitedError (+50 more)

### Community 47 - "promisedRequirementTest.py"
Cohesion: 0.06
Nodes (32): MesosAgentThread, MesosMasterThread, MesosTestSupport, MesosThread, Mixin for test cases that need a running Mesos master and agent on the local…, Gracefully stop a process on a timeout, given the Popen object for the process., AbstractPromisedRequirementsTest, _follower() (+24 more)

### Community 48 - "log_bindings"
Cohesion: 0.08
Nodes (35): AnyINode, is_any_url(), Decide if a string is a URI like http:// or file://. Otherwise it might be a…, clone_metadata(), drop_if_missing(), get_inode_nonexistent(), get_inode_virtualized_value(), get_shared_fs_path() (+27 more)

### Community 49 - "Base"
Cohesion: 0.07
Nodes (37): Base, Binding, Decl, Directory, File, SourceNode, SourcePosition, ensure_null_inodes_are_nullable() (+29 more)

### Community 50 - "deferredFunctionTest.py"
Cohesion: 0.09
Nodes (32): _deferredFunctionRunsWithFailuresFn(), _deleteFile(), _deleteMethods, options(), fixture, Namespace, Path, skipif (+24 more)

### Community 51 - "cleanup_aws_resources.py"
Cohesion: 0.13
Nodes (25): contains_num_only_uuid(), contains_toil_test_patterns(), contains_uuid(), contains_uuid_with_underscores(), find_buckets_in_region(), find_buckets_to_cleanup(), find_iam_roles_in_region(), find_iam_roles_to_cleanup() (+17 more)

### Community 52 - "TestUtils"
Cohesion: 0.12
Nodes (16): A convenience wrapper around subprocess.check_call that logs the command before…, system(), correctSort(), Any, fixture, Path, Popen, Tests the status and stats commands of the toil command line utility using the… (+8 more)

### Community 53 - "Any"
Cohesion: 0.12
Nodes (9): Any, Add a function as a child job. :param fn: Function to be run as a child job…, Add a function as a follow-on job. :param fn: Function to be run as a follow-on…, Add a job function as a child job. See :class:`toil.job.JobFunctionWrappingJob`…, Creates an encapsulated Toil job function with unfulfilled promised resource…, Instantiate this Promise., Initialize this Promised Requirement. :param valueOrCallable: A single Promise…, Return PromisedRequirement value. (+1 more)

### Community 54 - "WDLKubernetesClusterTest"
Cohesion: 0.29
Nodes (4): timeout, Ensure WDL works on the Kubernetes batchsystem., Test that a wdl workflow works on a kubernetes cluster. Launches a cluster with…, WDLKubernetesClusterTest

### Community 55 - "S3Test"
Cohesion: 0.29
Nodes (4): Confirm the workarounds for us-east-1., Test bucket creation for us-east-1., Test getting bucket location for a bucket we don't own., S3Test

### Community 56 - ".__init__"
Cohesion: 0.09
Nodes (7): HelloWorld, HelloWorld, Job initializer. This method must be called by any overriding constructor.…, HelloWorld, HelloWorld, HelloWorld, HelloWorld

### Community 57 - "serverTest.py"
Cohesion: 0.05
Nodes (36): MultiprocessingTaskRunner, Version of TaskRunner that just runs tasks with Multiprocessing. Can't use…, Cancel the task with the given ID., Return True if the task has not yet failed, and False otherwise. Returns True…, Returns True if the task has not yet stopped, and False otherwise., AbstractStateStoreTest, AWSStateStoreTest, BucketUsingTest (+28 more)

### Community 58 - "GCEProvisioner"
Cohesion: 0.08
Nodes (14): GCEProvisioner, Get the credentials from the file specified by GOOGLE_APPLICATION_CREDENTIALS., Try a few times to terminate all of the instances in the group., Implements a Google Compute Engine Provisioner using libcloud., Set up the credentials on the worker., Monkey patch to gce.py in libcloud to allow disk and images to be specified.…, Read the cluster settings from the instance, which should be the leader. See…, ClusterCombinationNotSupportedException (+6 more)

### Community 59 - "toil/lib/aws/__init__.py"
Cohesion: 0.07
Nodes (42): BaseApplication, get_aws_zone_from_boto(), get_aws_zone_from_environment(), get_aws_zone_from_environment_region(), get_aws_zone_from_metadata(), get_current_aws_region(), get_current_aws_zone(), Get the AWS zone from the Boto3 config file or from AWS_DEFAULT_REGION, if it… (+34 more)

### Community 60 - "TestJobService"
Cohesion: 0.09
Nodes (19): fnTest(), Any, Path, skipif, timeout, Tests the creation of a Job.Service with random failures of the worker, making…, Tests the creation of a Job.Service, creating a chain of services and accessing…, Tests the creation of a Job.Service, creating parallel chains of services and… (+11 more)

### Community 61 - "Appliance"
Cohesion: 0.07
Nodes (17): Self, Appliance, ApplianceTestSupport, LeaderThread, make_tests(), ProxyConnectionError, Any, BaseException (+9 more)

### Community 62 - "MesosBatchSystem"
Cohesion: 0.03
Nodes (40): Queue, Scheduler, Internal function. Should not be called outside this class. Make a job name…, Internal function. Should not be called outside this class. Create, if not…, MesosBatchSystem, _ArgumentGroup, ArgumentParser, Get the default IP/hostname and port that we will look for Mesos at. (+32 more)

### Community 63 - "needs_docker"
Cohesion: 0.11
Nodes (22): conformance, docker_cuda, kubernetes, local_cuda, docker, online, Subtests, timeout (+14 more)

### Community 64 - "tasks.py"
Cohesion: 0.08
Nodes (24): Celery, create_celery_app(), download_file_from_internet(), download_file_from_s3(), get_file_class(), get_iso_time(), link_file(), Return the current time in ISO 8601 format. (+16 more)

### Community 65 - "Any"
Cohesion: 0.08
Nodes (15): filter_skip_null(), Any, Init hook to set up member variables., Recursively filter out SkipNull objects from 'value'. :param name: Name of port…, Private implementation for recursively filtering out SkipNull objects from…, Create a child job with the correct resource requirements set., Return our properties dictionary., Add a child to our workflow. (+7 more)

### Community 66 - ".testConcurrencyWithDisk"
Cohesion: 0.05
Nodes (30): pslow, physicalDisk(), physicalMemory(), memoize, Calculate the total amount of physical memory, in bytes. >>> n =…, count(), get_omp_threads(), getCounters() (+22 more)

### Community 67 - "toilStats.py"
Cohesion: 0.08
Nodes (44): add_stats_options(), ColumnWidths, compute_column_widths(), decorate_subheader(), decorate_title(), get(), main(), pad_str() (+36 more)

### Community 68 - "wes_cwl_runner.py"
Cohesion: 0.09
Nodes (26): generate_attachment_path_names(), get_deps_from_cwltool(), main(), poll_run(), print_logs_and_exit(), Any, BytesIO, A modified version of the WESClient from the wes-service package that includes… (+18 more)

### Community 69 - "abstractBatchSystem.py"
Cohesion: 0.03
Nodes (46): AcquisitionTimeoutException, ABC, Exception, NamedTuple, Give the batch system an opportunity to connect directly to the message bus, so…, # FIXME: Return value should be a set (then also fix the tests), Initialize initial state of the object. :param toil.common.Config config:…, Give the batch system an opportunity to connect directly to the message bus, so… (+38 more)

### Community 70 - ".prepareSbatch"
Cohesion: 0.11
Nodes (12): any_option_detector(), option_detector(), parse_slurm_time(), _ArgumentGroup, ArgumentParser, Parse a Slurm-style time duration to a number of seconds. Slurm supports the…, Select suitable slurm partition based on requirements. Checks state of…, Returns the sbatch command line to run to queue the job. (+4 more)

### Community 71 - "global_mutex"
Cohesion: 0.07
Nodes (24): Create a new DeferredFunctionManager, sharing state with other instances in…, Clean up our state on disk. We assume that the deferred functions we manage…, Look at the state of all jobs registered in the individual job state files, and…, Generator function that deserializes and yields the job state for every job on…, :param shutdown_info: The coordination directory., Make sure the coordination directory hasn't been deleted unexpectedly. Slurm…, collect_process_name_garbage(), ensure_filesystem_lockable() (+16 more)

### Community 72 - "Node"
Cohesion: 0.04
Nodes (34): MAT, MRT, compat_bytes_recursive(), Any, Convert a tree of objects over bytes to objects over strings., parse_iso_utc(), datetime, Like memoize, but guarantees that decorated function is only called once, even… (+26 more)

### Community 73 - "ToilBackend"
Cohesion: 0.10
Nodes (21): handle_errors(), Raised when the requested workflow version is not implemented., This decorator catches errors from the wrapped function and returns a JSON…, VersionNotImplementedException, Any, Response, WES backend implemented for Toil to run CWL, WDL, or Toil workflows. This class…, Make a new ToilBackend for serving WES. :param work_dir: Directory to download… (+13 more)

### Community 74 - "AbstractToilWESServerTest"
Cohesion: 0.06
Nodes (26): AbstractToilWESServerTest, skipif, timeout, Class for server tests that provides a self.app in testing mode., Fetch the run log for a given workflow., Make sure the run log for the given run is generated correctly. The workflow…, Report the log for the given workflow run., Take a URL that has a hostname and port, relativize it to the test Flask… (+18 more)

### Community 75 - "decode_directory"
Cohesion: 0.16
Nodes (13): DirectoryStructure, IO, Return a local absolute path for a file (no schema). Overwrites…, Test for file existence., Copy input files out of the global file store and update location and path.…, toilStageFiles(), download_structure(), get_from_structure() (+5 more)

### Community 76 - "trim_mounts_op_up"
Cohesion: 0.50
Nodes (4): Get the local bare path for a CWL file or directory, or None. :return: None if…, Remove subtrees of the CWL file or directory object tree that only have…, sniff_location(), trim_mounts_op_up()

### Community 77 - "TestSlurmMountRecovery"
Cohesion: 0.07
Nodes (6): call_sinfo(), FakeBatchSystem, Simulate asking for partition info from Slurm, Class that implements a minimal Batch System, needed to create a Worker (see…, Returns a dummy config for the batch system tests. We need a workflowID to be…, TestSlurmMountRecovery

### Community 79 - "checkDockerImageExists"
Cohesion: 0.10
Nodes (17): checkDockerImageExists(), parseDockerAppliance(), Attempt to check a url registryName for the existence of a docker image with a…, Derive parsed registry, image reference, and tag from a docker image string.…, DockerCheckTest, Tests checking whether a docker image exists or not., Image exists. This should pass., Image exists. This should pass. (+9 more)

### Community 80 - "dockstore.py"
Cohesion: 0.09
Nodes (30): Cost, ensure_valid_id(), get_metrics_url(), pack_single_task_metrics(), pack_workflow_metrics(), pack_workflow_task_set_metrics(), TypedDict, # TODO: Is this meant to be actual usage or amount provided? (+22 more)

### Community 81 - "ToilDocumentationTest"
Cohesion: 0.12
Nodes (3): timeout, Tests for scripts in the toil tutorials., ToilDocumentationTest

### Community 82 - ".get_dependencies"
Cohesion: 0.12
Nodes (12): for_each_node(), Make a graph for analyzing a set of workflow nodes., Map multiple IDs for what we consider the same node to one ID. This…, Return True if a node represents a WDL declaration, and false otherwise., Get all the nodes that a node depends on, recursively (into the node if it has…, Get all the nodes that a node depends on, transitively., Get a topological order of the nodes, based on their dependencies., Get all the workflow node IDs that have no dependents in the graph. (+4 more)

### Community 83 - "lib.py"
Cohesion: 0.16
Nodes (25): announce(), check_to_cache(), complain(), file_link(), get_current_commit(), get_hostname(), in_acceptable_environment(), is_rebase() (+17 more)

### Community 84 - ".getLogger"
Cohesion: 0.14
Nodes (11): JSONDatagramHandler, Any, Logger, LogRecord, Get the logger that logs real-time to the leader. Note that if the returned…, Send logging records over UDP serialized as JSON. They have to fit in a single…, Actually, encode the record as bare JSON instead., Metaclass for RealtimeLogger that lets add logging methods. Like… (+3 more)

### Community 85 - "GridEngineThread"
Cohesion: 0.21
Nodes (3): GridEngineThread, A very simple script generator that just wraps the command given; for now this…, Determines PBS/Torque version via pbsnodes

### Community 86 - "RuntimeError"
Cohesion: 0.13
Nodes (10): JobType, RuntimeError, Make sure this JobDescription is not newer than a prospective new version of…, Ensure that Job.__init__() has been called by any subclass __init__(). This…, Called whenever the job graphs of this job and the other job may have been…, Add a childJob to be run as child of this job. Child jobs will be run directly…, Add a follow-on job. Follow-on jobs will be run after the child jobs and their…, Save the execution data for just this job to the JobStore, and fill in the… (+2 more)

### Community 87 - "call_command"
Cohesion: 0.12
Nodes (10): GridEngineThread, Grid Engine-specific AbstractGridEngineWorker methods, Get job exist code, checking both qstat and qacct. Return None if still…, GridEngineThread, Helper functions for getJobExitCode and to parse the bjobs status record, Parse records from bjobs json type output :params bjobs_output_str: stdout of…, Parse the maximum memory from job. :param jobID: ID number of the job, LSF specific GridEngineThread methods. (+2 more)

### Community 88 - "LastProcessStandingArena"
Cohesion: 0.11
Nodes (18): OSError, LastProcessStandingArena, Class that lets a bunch of processes detect and elect a last process standing.…, BaseSafeLockingTest, Path, safe_unlock_and_close should swallow the error and still close the fd., Tests safe_lock and safe_unlock_and_close behavior when fcntl raises ENOLCK…, Tests safe_lock and safe_unlock_and_close behavior when fcntl raises EIO (Ceph… (+10 more)

### Community 89 - "FileJobStore"
Cohesion: 0.03
Nodes (59): CacheError, InvalidSourceCacheError, Exception, # TODO: give other people time to finish their in-progress, # TODO: this empty file could leak if we die now..., # TODO: work out if that will never happen somehow., # TODO: Maybe stream from cache even when not required for consistency?, # TODO: should we just let other jobs and the cache keep the file until (+51 more)

### Community 90 - "test_ec2.py"
Cohesion: 0.13
Nodes (21): aws_marketplace_flatcar_ami_search(), feed_flatcar_ami_release(), _fetch_flatcar_feed(), flatcar_release_feed_ami(), get_flatcar_ami(), BaseClient, RuntimeError, Yield AMI IDs for the given architecture from the Flatcar release feed. :param… (+13 more)

### Community 91 - "ensure_file_imported"
Cohesion: 0.12
Nodes (18): CWLDirectoryType, CWLFileType, MutableSequence, ensure_file_imported(), ensure_no_collisions(), extract_file_uri_once(), filtered_secondary_files(), import_file_through_cache() (+10 more)

### Community 92 - "._parseResource"
Cohesion: 0.10
Nodes (18): ParseableAcceleratorRequirement, ParseableDivisibleResource, ParseableFlag, ParseableIndivisibleResource, ParseableRequirement, ParsedRequirement, Any accelerators, such as GPUs, that are needed., Memory, core and disk requirements are specified identically to as in \… (+10 more)

### Community 93 - "version_template.py"
Cohesion: 0.11
Nodes (22): check_cwltool_version(), Check if the installed cwltool version matches Toil's expected version. A…, logProcessContext(), cacheTag(), currentCommit(), dirty(), distVersion(), dockerTag() (+14 more)

### Community 94 - "._connect"
Cohesion: 0.11
Nodes (12): FTP, Any, IO, Parse an FTP url into hostname, username, password, and path :param url:…, Connect to an FTP server. Handles authentication. :param url: FTP url :return:…, Grab the cached credentials :param desired_host: FTP hostname :return:…, Get the size of an FTP object :param fn: FTP url :return: Size of object, FTP object to handle FTP connections. By default, connect over FTP with TLS.… (+4 more)

### Community 95 - "toil/__init__.py"
Cohesion: 0.14
Nodes (18): ImageNotFound, ApplianceImageNotFound, _check_custom_bash_cmd(), checkDockerSchema(), customDockerInitCmd(), customInitCmd(), lookupEnvVar(), Return the custom command set by the ``TOIL_CUSTOM_DOCKER_INIT_COMMAND``… (+10 more)

### Community 96 - ".session"
Cohesion: 0.14
Nodes (12): ServiceResource, Session, _new_boto3_session(), BaseClient, Get the Boto3 Session to use for the given region., Get the Boto3 Resource to use with the given service (like 'ec2') in the given…, Get the Boto3 Client to use with the given service (like 'ec2') in the given…, Get a Boto 3 resource for a particular AWS service, usable by the current… (+4 more)

### Community 97 - "threading.py"
Cohesion: 0.04
Nodes (38): add_all_batchsystem_options(), _ArgumentGroup, ArgumentParser, # TODO: Move this to Slurm specifically., Call set_option for all the options for the given named batch system, or all…, set_batchsystem_options(), add_batch_system_factory(), get_batch_system() (+30 more)

### Community 98 - "Submission"
Cohesion: 0.12
Nodes (19): JobAttemptSummary, Data class holding summary information for a workflow attempt. Helpfully…, Data class holding summary information for a job attempt within a known…, get_parsed_trs_spec(), job_execution_id(), Class holding a package of information to submit to Dockstore, and the…, Create a new empty submission., Add a workflow attempt to the submission. May raise an exception if the… (+11 more)

### Community 99 - ".add_toil_service"
Cohesion: 0.10
Nodes (11): RuntimeError, Return Bash commands that set up the Kubernetes cluster autoscaler for…, Return the Kubernetes cloud provider (for example, 'aws'), to pass to the…, Add services to configure as a Kubernetes leader, if Kubernetes is already set…, Add services to configure as a Kubernetes worker, if Kubernetes is already set…, Return the text (not bytes) user data to pass to a provisioned node. If leader-…, Write a file to a physical storage system that is accessible to the leader and…, Get the maximum number of bytes that can be passed as the user data during node… (+3 more)

### Community 100 - "lsfHelper.py"
Cohesion: 0.11
Nodes (26): Make a bsub commandline to execute. params: cpu: number of cores needed mem:…, apply_bparams(), apply_conf_file(), apply_lsadmin(), check_lsf_json_output_supported(), find(), find_first_match(), get_conf_file() (+18 more)

### Community 101 - "Config"
Cohesion: 0.06
Nodes (18): memoize, Get the directory where the backing batch system should save its logs. Only…, Format path for batch system standard output/error and other files generated by…, Get a glob string that will match all file paths generated by…, Config, Class to represent configuration operations for a toil workflow run., Return a path to a writable directory under which per-workflow directories…, After options are set, prepare for initial start of workflow. (+10 more)

### Community 102 - "AbstractJobStore"
Cohesion: 0.02
Nodes (77): NotImplementedError, Issues a job with the specified command to the batch system and returns a…, Make the file associated with fileStoreID available locally. If mutable is…, Update the status of the job on the disk. May bump the version number of the…, Blocks while startCommit is running. This function is called by this job's…, Perform setup work that requires the JobStore. Called by the Job saving logic…, Create a context manager that yields a file handle to the log file. Assumes…, Setup flag files. When a ServiceJobDescription first meets the JobStore, it… (+69 more)

### Community 103 - "ValueError"
Cohesion: 0.07
Nodes (21): parse_slurm_option_value(), Parse a Slurm option value in either ``--opt=value`` or ``--opt value`` form.…, Return a copy of this object with the given requirement scaled up or down. Only…, _bail(), decrypt(), encrypt(), decrypt(), encrypt() (+13 more)

### Community 104 - "makeRootJob"
Cohesion: 0.12
Nodes (21): extract_workflow_inputs(), get_options(), import_workflow_inputs(), makeRootJob(), path_to_loc(), DirectoryContents, Namespace, T (+13 more)

### Community 105 - "JobStatus"
Cohesion: 0.12
Nodes (22): JobStatus, Records the status of a job. When exit_code is -1, this means the job is either…, MalformedRequestException, # TODO: make this a typed dict with all the WES task log field names and their…, Raised when the request is malformed., DataDict, FilesDict, parse_workflow_manifest_file() (+14 more)

### Community 106 - "ToilRestartException"
Cohesion: 0.13
Nodes (12): Any, Invoke a Toil workflow with the given job as the root for an initial run. This…, Restarts a workflow that has been interrupted. :return: The root job's return…, Create an instance of the batch system specified in the given config. :param…, Determine the user script, save it to the job store and inject a reference to…, Set the environment variables required by the job store and those passed on…, Put the environment in a globally accessible pickle file., Download all jobs in the current job store into self.jobCache. (+4 more)

### Community 107 - "leader.py"
Cohesion: 0.07
Nodes (25): DeadlockException, Exception, Exception thrown by the Leader or BatchSystem when a deadlock is encountered…, Stringify the exception, including the message., # TODO: couldn't more jobs have started since we polled the, # TODO: When a job stops running but has yet to be collected from the, # TODO: make this update fast enough to put it in the progress, # TODO: Give other components a chance to connect to the bus before (+17 more)

### Community 108 - "apiDockerCall"
Cohesion: 0.07
Nodes (40): count_amd_gpus(), count_nvidia_gpus(), get_host_accelerator_numbers(), get_individual_local_accelerators(), memoize, Return the number of nvidia GPUs seen by nvidia-smi, or 0 if it is not working., # TODO: Parse each gpu > product_name > text content and convert to some, Return the number of amd GPUs seen by rocm-smi, or 0 if it is not working.… (+32 more)

### Community 109 - ".restartCheckpoint"
Cohesion: 0.10
Nodes (10): Get an iterator over all child, follow-on, and service job IDs., Get an iterator over all child, follow-on, and chained, inherited successor job…, Returns True if we have a job body associated, and False otherwise., Get the information needed to load the job body. :returns: a file store ID (or…, Return the collection of job IDs for the successors of this job that are ready…, Remove all references to successor and service jobs., Check if the subtree is done. :returns: True if the job appears to be done, and…, Save a body checkpoint into self.checkpoint (+2 more)

### Community 110 - "AtomicFileCreate"
Cohesion: 0.13
Nodes (16): getDirSizeRecursively(), getNodeID(), StrPath, This method will return the cumulative number of bytes occupied by the files on…, Return unique ID of the current node (host). The resulting string will be…, atomic_install(), atomic_tmp_file(), AtomicFileCreate() (+8 more)

### Community 111 - "WESBackend"
Cohesion: 0.12
Nodes (11): Any, A class to represent a GA4GH Workflow Execution Service (WES) API backend.…, Map an operationId defined in the OpenAPI or swagger yaml file to a function.…, Get information about the Workflow Execution Service. GET /service-info, List the workflow runs. GET /runs, Run a workflow. This endpoint creates a new workflow run and returns a `RunId`…, Get detailed info about a workflow run. GET /runs/{run_id}, Cancel a running workflow. POST /runs/{run_id}/cancel (+3 more)

### Community 112 - "history_submission.py"
Cohesion: 0.10
Nodes (24): KeyType, get_default_config_path(), memoize, Get the default path where the Toil configuration file lives. The file at the…, Set the given top-level key to the given value in the given YAML config file.…, update_config(), ask_user_about_publishing_metrics(), dialog_tkinter() (+16 more)

### Community 113 - "options/common.py"
Cohesion: 0.08
Nodes (28): Action, ParseableSingleAcceleratorRequirement, _ArgumentGroup, ArgumentParser, Parse a job store locator to a type string and the data needed for that…, Turn a job store locator into one that will work from any directory and always…, parse_accelerator(), Parse an AcceleratorRequirement specified by user code. Supports formats like:… (+20 more)

### Community 114 - "GridEngineThread"
Cohesion: 0.15
Nodes (10): GridEngineThread, Any, JobTuple, Connect to HTCondor Schedd and yield a Schedd object. You can only use it…, Ping the scheduler, or fail if it persistently cannot be contacted., Get the HTCondor scheduler to connect to, or None for the local machine.…, Open a connection to the htcondor schedd. Assumes that the necessary lock is…, Escape a string by doubling up all single and double quotes. This is used for… (+2 more)

### Community 115 - "slurm_mount_recovery.py"
Cohesion: 0.09
Nodes (25): batch_logs_indicate_storage_failure(), build_scontrol_argv(), env_bool(), env_csv(), _expand_slurm_bracket_token(), is_fatal_storage_oserror(), parse_scontrol_job_lines(), parse_slurm_nodelist() (+17 more)

### Community 116 - "generate_default_job_store"
Cohesion: 0.11
Nodes (15): generate_default_job_store(), generate_locator(), JobStoreUnavailableException, NoAvailableJobStoreException, Exception, RuntimeError, Raised when a particular type of job store is requested but can't be used., Generate a random locator for a job store of the given type. Raises an… (+7 more)

### Community 117 - "SortTest"
Cohesion: 0.10
Nodes (19): copySubRangeOfFile(), down(), getMidPoint(), main(), makeFileToSort(), merge(), Merges the two files and places them in the output., Sorts the given file. (+11 more)

### Community 118 - "._get_blob_from_url"
Cohesion: 0.16
Nodes (9): Blob, Client, permission_error_reporter(), IO, memoize, ParseResult, Produce a client for Google Storage with the highest level of access we can…, Gets the blob specified by the url. caution: makes no api request. blob may not… (+1 more)

### Community 119 - "TestJob"
Cohesion: 0.06
Nodes (30): fn2Test(), Exception, Path, Service, Subtests, timeout, xfail, Make sure jobs retry with exponential backoff. (+22 more)

### Community 120 - "skip"
Cohesion: 0.20
Nodes (9): Decide what to automatically generate documentation for., skip(), gridengine, lsf, mesos, slurm, Run the CWL 1.0 conformance tests in various environments., TestCWLv10Conformance (+1 more)

### Community 121 - "worker.py"
Cohesion: 0.03
Nodes (50): IO, Return the directory where coordination files should be located for this…, safeUnpickleFromStream(), Any, Create a concreate FileStore., Read and write dill-ed state dictionaries from/to a file into a namespace., This is a context manager that state file and reads it into an object that is…, Load the state of the cache from the state file. :param fileName: Path to the… (+42 more)

### Community 122 - "toilDebugTest.py"
Cohesion: 0.14
Nodes (18): Return the full path to the venv Python on the leader., fetchFiles(), Path, Test toilDebugFile.fetchJobStoreFiles() symlinks., Test the toil debug-job command., Get a job store and the ID of a failing job within it., # TODO: This assumes a lot about the FileJobStore. Use the MessageBus instead?, Get a job store and the name of a failed job in it that actually wanted to use… (+10 more)

### Community 123 - "url.py"
Cohesion: 0.12
Nodes (13): CWLUnsupportedException, Exception, # TODO: why?, RuntimeError, # TODO: isn't this built in to Python 3 now?, UnimplementedURLException, aws_job_store_factory(), file_job_store_factory() (+5 more)

### Community 124 - "slurmSortTest.py"
Cohesion: 0.12
Nodes (18): skipUnless, Find the path to the given entry point that *should* work on a worker.…, resolveEntryPoint(), _partition_has_alternate(), Any, End-to-end Slurm workflows for mount recovery and partition handling., Run a workflow entry point on Slurm and return captured leader log text. Cleans…, Merge-sort smoke test on Slurm (same pattern as SortTest.testFileSingle). (+10 more)

### Community 125 - "InstanceConfiguration"
Cohesion: 0.13
Nodes (9): InstanceConfiguration, Allows defining the initial setup for an instance and then turning it into an…, Make a file on the instance with the given filesystem, mode, and contents. See…, Make a systemd unit on the instance with the given name (including .service),…, Authorize the given bare, encoded RSA key (without "ssh-rsa")., Return an Ignition configuration describing the desired config., Get the base configuration for both leader and worker instances for all cluster…, Add a service to prepare and mount local scratch volumes. (+1 more)

### Community 126 - "WorkflowStateStore"
Cohesion: 0.17
Nodes (7): Slice of a state store for the state of a particular workflow., Wrap the given state store for access to the given workflow's state., Get the given item of workflow state., Set the given item of workflow state., Read a value from a local cache, without checking the actual backend., Write a value to a local cache, without modifying the actual backend., WorkflowStateStore

### Community 127 - "WorkflowStateMachine"
Cohesion: 0.13
Nodes (12): Class for managing the WES workflow state machine. This is the authority on the…, Set the state to the given value, if a read does not show a terminal state…, Send an enqueue message that would move from UNKNOWN to QUEUED., Send an initialize message that would move from QUEUED to INITIALIZING., Send a run message that would move from INITIALIZING to RUNNING., Send a cancel message that would move to CANCELING from any non-terminal state., Send a canceled message that would move from CANCELING to CANCELED., Send a complete message that would move from RUNNING to COMPLETE. (+4 more)

### Community 128 - "UserDefinedJobArgTypeTest"
Cohesion: 0.15
Nodes (10): Foo, JobClass, jobFunction(), main(), Test for issue #423 (Toil can't unpickle classes defined in user scripts) and…, Test with first job being a function, Test with first job being an instance of a class, Test with first job being a function defined in __main__ (+2 more)

### Community 129 - "plugins.py"
Cohesion: 0.15
Nodes (15): PluginType, get_plugin(), get_plugin_names(), _load_all_plugins(), _plugin_name_prefix(), Any, memoize, Load all the plugins of the given type that are installed. (+7 more)

### Community 130 - "Info"
Cohesion: 0.06
Nodes (21): Info, Popen, Record for a running job. Stores the start time of the job, the Popen object…, Be the "daddy" thread. Our job is to look at jobs from the input queue. If a…, Stop the given child processes and all their children. Does not reap them.…, Stop the given child processes and all their children. Blocks until the…, Wait for the process groups to be killed. Blocks until the processes are gone…, See if any children represented in the given dict from PID to Popen object have… (+13 more)

### Community 131 - "toil_backend.py"
Cohesion: 0.10
Nodes (17): OperationForbidden, Exception, :param options: A list of default engine options to use when executing a…, Raised when the requested run ID is not found., Raised when the requested workflow is not in the expected state., Raised when the request is forbidden., Raised when an internal error occurred during the execution of the workflow., WorkflowConflictException (+9 more)

### Community 132 - "ToilWorkflow"
Cohesion: 0.10
Nodes (14): TaskLog, TextIO, Get a context manager for either a stream for the given file from the…, Return True if the workflow run exists., Set up necessary directories for the run., Clean directory and files related to the run., Return a collection of output files that this workflow generated., Return the given relative path from self.scratch_dir, if it is a file, and None… (+6 more)

### Community 133 - ".__init__"
Cohesion: 0.13
Nodes (5): Create a ServiceJobDescription to describe a ServiceHostJob., Create a CheckpointJobDescription to describe a checkpoint job., :param callable userFunction: The function to wrap. It will be called with…, :param d: Sequence of dictionaries to merge, Setup importing files on a worker. :param filenames: List of file URIs to…

### Community 134 - "sortTest.py"
Cohesion: 0.18
Nodes (17): copySubRangeOfFile(), down(), getMidPoint(), main(), makeFileToSort(), merge(), Merges the two files and places them in the output., Sorts the given file. (+9 more)

### Community 135 - "CachingFileStore"
Cohesion: 0.06
Nodes (22): CachingFileStore, Any, Cursor, Our current job that was using oldJobReqs space has finished. We need to record…, A cache-enabled file store. Provides files that are read out as symlinks or…, This context manager decorated method allows cache-specific operations to be…, Create a hardlink or symlink from the given path in the cache to the given…, Run in a thread to actually commit the current job. (+14 more)

### Community 136 - "stress.py"
Cohesion: 0.20
Nodes (6): HelloWorldFollowOn, HelloWorldJob, LongTestFollowOn, LongTestJob, main(), touchFile()

### Community 137 - "ec2nodes.py"
Cohesion: 0.11
Nodes (18): BlockDeviceMappingTypeDef, download_region_json(), InstanceType, is_number(), parse_memory(), parse_storage(), Any, Determines if a unicode string (that may include commas) is a number. :param s:… (+10 more)

### Community 138 - "UserNameUnvailableTest"
Cohesion: 0.22
Nodes (4): Make sure we can get something for a user name when user names are not…, Make sure we can get something for a user name when user name fetching is…, UserNameUnvailableTest, UserNameVeryBrokenTest

### Community 139 - "ServiceManager"
Cohesion: 0.08
Nodes (16): Fetch a ready client, waiting as needed. :param float maxWait: Time in seconds…, Fetch a client whos services failed to start. :param float maxWait: Time in…, Fetch a service job that is ready to start. :param maxWait: Time in seconds to…, Stop all the given service jobs. :param services: Service jobStoreIDs to kill…, Return true if the service job has not been told to terminate. :param…, Return true if the service job has started and is active. :param service:…, Check on the service manager thread. :raise RuntimeError: If the underlying…, Terminate worker threads cleanly; starting and killing all service threads.… (+8 more)

### Community 140 - "jobServiceTest.py"
Cohesion: 0.20
Nodes (10): Event, Creates one service and one accessing job, which communicate with two files to…, Creates a chain of services and accessing jobs, each paired together., Creates multiple chains of services and accessing jobs., Writes a random integer iinto the inJobStoreFileID file, then tries 10 times…, serviceAccessor(), serviceTest(), serviceTestParallelRecursive() (+2 more)

### Community 141 - "TestImportExportFile"
Cohesion: 0.35
Nodes (6): create_file(), Namespace, Path, Subtests, Ensures that uploaded files preserve their file permissions when they are…, TestImportExportFile

### Community 142 - "toil-sort-example.py"
Cohesion: 0.18
Nodes (16): copy_subrange_of_file(), down(), get_midpoint(), main(), make_file_to_sort(), merge(), Copies the range (in bytes) between fileStart and fileEnd to the given output…, Finds the point in the file to split. Returns an int i such that fileStart <= i… (+8 more)

### Community 143 - ".read_from_url"
Cohesion: 0.16
Nodes (8): IO, ParseResult, Read the given URL and write its content into the given writable stream. Raises…, Read from the given URI. Raises FileNotFoundError if the URL doesn't exist. Has…, Reads the contents of the object at the specified location and writes it to the…, Get a stream of the object at the specified location. Raises FileNotFoundError…, Reads the contents of the given readable stream and writes it to the object at…, Returns True if the url access implementation supports the URL's scheme. :param…

### Community 144 - "._file_maker"
Cohesion: 0.20
Nodes (8): Any, Path, Get a function that mints fresh files that can be uploaded to a job store., Check that, when using hints, a person would be able to find the file., Verify that deleting a hinted file and creating a new one with the same hints…, Hint components that would collide with the layout's reserved directory names…, Adding a hint that's the same as the disambiguating numbers needs to not…, A hint list that is a prefix of another hint list whose extension is purely…

### Community 145 - "setter"
Cohesion: 0.18
Nodes (6): setter, Get the number of tries remaining. The try count set on the JobDescription, or…, The maximum number of bytes of disk the job will require to run., The maximum number of bytes of memory the job will require to run., The number of CPU cores required., Whether the job can be run on a preemptible node.

### Community 146 - "toilContextManagerTest.py"
Cohesion: 0.16
Nodes (7): getTempFile(), get_temp_file(), Return a string representing a temporary file, that must be manually deleted., childFn(), FollowOn, HelloWorld, ToilContextManagerTest

### Community 147 - "directory.py"
Cohesion: 0.14
Nodes (20): check_directory_dict_invariants(), directory_contents_items(), directory_item_exists(), directory_items(), encode_directory(), get_directory_contents_item(), get_directory_item(), DirectoryContents (+12 more)

### Community 148 - "RealtimeLogger"
Cohesion: 0.22
Nodes (6): BaseException, TracebackType, Provide a logger that logs over UDP to the leader. To use in a Toil job, do:…, Stop the server on the leader., Create a context manager that starts up the UDP server. Should only be invoked…, RealtimeLogger

### Community 149 - "DirectoryResource"
Cohesion: 0.18
Nodes (7): DirectoryResource, BytesIO, IO, Download this resource from its URL to a file on the local system. This method…, Returns a readable file-like object for the given path. If the path refers to a…, Download this resource from its URL to the given file object. :type dstFile:…, A resource read from a directory on the leader. The URL will point to a ZIP…

### Community 150 - "MemoryStateCache"
Cohesion: 0.11
Nodes (11): MemoryStateCache, MemoryStateStore, An in-memory place to store workflow state., Make a new in-memory state cache., Get a key value from memory., Set or clear a key value in memory., Set up the AbstractStateStore and its cache., An in-memory place to store workflow state, for testing. Inherits from… (+3 more)

### Community 151 - ".test_symlink_read_control"
Cohesion: 0.20
Nodes (3): Test that imported files are symlinked when when expected, Test that files are read by symlink when expected, Test writing log files.

### Community 152 - "AutoDeploymentTest"
Cohesion: 0.18
Nodes (8): AutoDeploymentTest, Test whether auto-deployment works with a virtualenv in which jobs are defined…, Tests various auto-deployment scenarios. Using the appliance, i.e. a docker…, Test encapsulated, function-wrapping jobs where the function arguments…, Ensure that the following DAG succeeds:: ┌───────────┐ │ Root (W1) │…, Creates an appliance cluster with a virtualenv at './venv' on the leader and a…, Ensure that the following DAG succeeds:: ┌───────────┐ │ Root (W1) │…, Test whether auto-deployment works on restart.

### Community 153 - "TestCleanWorkDir"
Cohesion: 0.28
Nodes (7): Namespace, Path, Tests testing :class:toil.fileStores.abstractFileStore.AbstractFileStore, Runs toil with the specified job and cleanWorkDir setting. expectError…, tempFileTestErrorJob(), tempFileTestJob(), TestCleanWorkDir

### Community 154 - "TestWDLToilBench"
Cohesion: 0.12
Nodes (13): Array, String, Tests for Toil's MiniWDL-based implementation that don't run workflows., Parse pseudo-WDL for testing whitespace removal., Test to make sure that we pick sensible but non-colliding directories to put…, Test to make sure the disk parsing is correct, TestWDLToilBench, choose_human_readable_directory() (+5 more)

### Community 155 - "create_current_submission"
Cohesion: 0.33
Nodes (5): BaseException, TracebackType, Clean up after a workflow invocation. Depending on the configuration, delete…, create_current_submission(), Make a package of data about the current workflow attempt to send in. Useful if…

### Community 156 - "slurm.py"
Cohesion: 0.04
Nodes (45): AbstractGridEngineBatchSystem, ExceededRetryAttempts, GridEngineThreadException, Exception, # TODO: Why do we need a lock for this? We have the GIL., # TODO: Note that this currently stores a tuple of (batch system, Returns exit codes and possibly exit reasons for a list of jobs, or None if…, Kills the given jobs, represented as Job ids, then checks they are dead by… (+37 more)

### Community 157 - "TestSafeFileInterleaving"
Cohesion: 0.13
Nodes (12): fixture, Path, timeout, Tests that verify locking correctness through deterministic interleavings. Each…, Set up test file path for each test., Context manager that installs all checkpointer patches. Uses ExitStack to…, Verify that a reader cannot proceed while a writer holds the exclusive lock.…, Verify that a writer cannot proceed while a reader holds a shared lock. (+4 more)

### Community 158 - "ToilTest"
Cohesion: 0.05
Nodes (27): parse_mem_and_cmd_from_output(), Use regex to find "MAX MEM" and "Command" inside of an output., build_tag_dict_from_env(), LSFHelperTest, lsfHelper.py shouldn't need a batch system and so the unit tests here should…, memoize, Pick an appropriate AWS region. Use us-west-2 unless running on EC2, in which…, Query this instance's metadata to determine in which availability zone it is… (+19 more)

### Community 159 - "utilsTest.py"
Cohesion: 0.17
Nodes (11): printUnicodeCharacter(), # TODO: Run these for the other clouds., # TODO: we need to reach into the FileJobStore's files and delete this, Runs child job with same resources as self in an attempt to chain the jobs on…, RunTwoJobsPerWorker, create_summary(), get_stats(), process_data() (+3 more)

### Community 160 - "conf.py"
Cohesion: 0.19
Nodes (11): fetch_parent_dir(), Returns a parent directory, n places above the input filepath. Equivalent to…, # TODO: It's not clear that the contidions used here are a good idea. Why, setup(), build_full_toctree(), get_rendered_toctree(), html_page_context(), Event handler for the html-page-context signal. Modifies the context directly.… (+3 more)

### Community 161 - "._createJobStore"
Cohesion: 0.18
Nodes (3): This test is meant to cover multi-part uploads in the AWSJobStore but it…, Test the reading and writing of large files., Ensure that the command line configurations are successfully loaded and stored.…

### Community 162 - "awsBatch.py"
Cohesion: 0.06
Nodes (25): Any, # TODO: Use a global AWSConnectionManager so we can share a client, # TODO: Deduplicate with Kubernetes batch system., # TODO: Keep this in sync with the Dockerfile., # TODO: retry!, Internal function. Should not be called outside this class. Get the time that…, Internal function. Should not be called outside this class. Get the exit code…, # TODO: How do we tolerate it not existing anymore? (+17 more)

### Community 163 - "executor"
Cohesion: 0.29
Nodes (4): executor(), Main function of the _toil_contained_executor entrypoint. Runs inside the Toil…, Prepares this system for the downloading and lookup of resources. This method…, Remove all downloaded, localized resources.

### Community 164 - "resumabilityTest.py"
Cohesion: 0.27
Nodes (9): badChild(), chaining_parent(), goodChild(), parent(), Set up a failing job to chain to., Fails the first time it's run, succeeds the second time., https://github.com/BD2KGenomics/toil/issues/808, Set up a bunch of dummy child jobs, and a bad job that needs to be restarted as… (+1 more)

### Community 165 - "Toil Job Store Storage — Cost & Architecture Analysis"
Cohesion: 0.20
Nodes (9): Deployment context (Augmet / ParallelCluster), Executive summary, Implementation backlog (prefix-based S3 job store), Open inputs for refined estimates, Related code references, Revision history, S3 request cost estimate for frequent `--restart`, Toil Job Store Storage — Cost & Architecture Analysis (+1 more)

### Community 166 - "get_job_kind"
Cohesion: 0.20
Nodes (5): Convert to human-readable string. Given an int that may be or may be equal to a…, Count the number of cluster-allocateable GPUs we want to allocate for the given…, get_job_kind(), Return an identifying string for the job. The result may contain spaces.…, Gather any new, updated JobDescriptions from the batch system.

### Community 167 - "TestResource"
Cohesion: 0.27
Nodes (6): Path, Asserts that Toil enforces the user script to have a .py or .pyc extension…, Write a file with the given contents, and keep it on disk as long as the…, Test module descriptors and resources derived from them., tempFileContaining(), TestResource

### Community 168 - ".testGetStatusFailedCWLWF"
Cohesion: 0.24
Nodes (7): cwl, docker, MonkeyPatch, online, Test that ToilStatus.getStatus() behaves as expected with a failing CWL…, Test that ToilStatus.getStatus() behaves as expected with a successful CWL…, Test that ToilStatus.printJobLog() reads the log from a failed command without…

### Community 169 - "strtobool"
Cohesion: 0.08
Nodes (27): main(), # TODO: Remove these paths as typing is added and mypy conflicts are addressed., Make a human-readable string into a bool. Convert a string along the lines of…, strtobool(), glob(), StrPath, Walks through a directory and its subdirectories looking for files matching the…, add_wdl_options() (+19 more)

### Community 170 - ".run"
Cohesion: 0.15
Nodes (7): CWLOutputType, Instantiate a conditional expression. :param expression: Expression from the…, Extract the given key from the obj. If the object is a list, extract it from…, Gather all the outputs of the scatter., First apply linkMerge then pickValue if either present., Apply linkMerge operator to `values` object. :param values: result of step, Apply pickValue operator to `values` object. :param values: Intended to be a…

### Community 171 - "Contributor Covenant Code of Conduct"
Cohesion: 0.25
Nodes (7): Attribution, Contributor Covenant Code of Conduct, Enforcement, Our Pledge, Our Responsibilities, Our Standards, Scope

### Community 172 - "JuiceFS — performance, stability, and Toil fit"
Cohesion: 0.25
Nodes (8): How Toil uses a file job store, JuiceFS — performance, stability, and Toil fit, Performance — will it suffer vs EBS?, Recommended Toil flags on JuiceFS, Restart latency ordering (typical), Stability — will it suffer vs EBS?, Three-way comparison (performance & stability), When to choose JuiceFS over EBS or S3

### Community 173 - ".formatLogStream"
Cohesion: 0.29
Nodes (4): Make an exception to report failed jobs. :param job_store: The job store with…, IO, Given a stream of text or bytes, and the job name, job itself, or some other…, Takes a list of jobs, finds their log files, and prints them to the terminal.

### Community 174 - "history.py"
Cohesion: 0.11
Nodes (15): HistoryDatabaseSchemaTooNewError, RuntimeError, # TODO: When Dockstore can take job metrics alongside whole-workflow, Get the path at which the database we store history in lives., # TODO: Do a try-and-fall-back to avoid sending the table schema for, # TODO: Should we force workflow attempts to be reported on, Raised when we would write to the history database, but its schema is too new…, # TODO: Make name of this function less general? (+7 more)

### Community 175 - "._loadUserModule"
Cohesion: 0.29
Nodes (4): ModuleType, Imports and returns the module object represented by the given module…, Unpickles an object graph from the given file handle while loading symbols \…, Set the values for promises using the return values from this job's run()…

### Community 176 - ".killJobs"
Cohesion: 0.29
Nodes (4): Kills the given set of jobs and then sends them for processing. Returns the…, Check each issued job. If a job is running for longer than desirable issue a…, Check all the current job ids are in the list of currently issued batch system…, Process jobs that have gone awry.

### Community 177 - ".handle_injection_messages"
Cohesion: 0.25
Nodes (4): Become responsible for the given peak memory usage, in kibibytes. The memory…, Become responsible for the given CPU time. The CPU time will be treated as if…, Handle any data received from injected runtime code in the container., Handle a message file received from in-container injected code. Takes the host-…

### Community 178 - "environmentTest.py"
Cohesion: 0.20
Nodes (12): check_environment(), check_environment_repeatedly(), EnvironmentTest, main(), Namespace, Test to make sure that Toil's environment variable save and restore system…, Make a file in the file store that the leader can see., Fail if the test environment is wrong. (+4 more)

### Community 179 - ".test_cwl_toil_kill"
Cohesion: 0.29
Nodes (4): Path, Test "toil kill" on a CWL workflow with a 100 second sleep., Test "toil kill" on a CWL workflow with a 100 second sleep., Test "toil kill" on a CWL workflow with a 100 second sleep.

### Community 180 - "wdltoil_test.py"
Cohesion: 0.12
Nodes (17): fixture, # TODO: Should this move to the TRS/Dockstore tests file?, # TODO: enable test if nvidia-container-runtime and Singularity are installed…, # TODO: enable test if nvidia-container-runtime and Singularity are installed…, # TODO: Reduce memory requests with custom/smaller inputs., # TODO: Skip if node lacks enough memory., WDL conformance tests for Toil., Make sure a call completed or explain why it failed. (+9 more)

### Community 181 - "Notes for AI Assistants"
Cohesion: 0.29
Nodes (6): Code Style, Development Environment, Notes for AI Assistants, Running Individual WDL Spec Unit Tests, Running Make Targets (mypy, tests, etc.), Running Tests

### Community 182 - ".forModule"
Cohesion: 0.18
Nodes (9): Any, args, kwargs, P, Capture the given callable and arguments as an instance of this class. :param…, inVirtualEnv(), Test if we are inside a virtualenv or Conda virtual environment., Exception (+1 more)

### Community 183 - "Production job counts and S3 restart latency"
Cohesion: 0.29
Nodes (7): Estimated S3 restart time (leader phase only), How Toil restart loads an S3 job store, Implications by run size, Job count distribution (124 production runs), Mitigations (no Toil code change), Observed restart pattern, Production job counts and S3 restart latency

### Community 184 - ".testAWSProvisionerUtils"
Cohesion: 0.29
Nodes (5): rsync, aws_s3, timeout, Runs a number of the cluster utilities in sequence. Launches a cluster with…, Test that the job store is only destroyed when we observe a successful workflow…

### Community 185 - "uploadFile"
Cohesion: 0.33
Nodes (7): fileSizeAndTime(), Any, IO, Upload a readable object to s3, using multipart uploading if applicable. :param…, Uploads a file to s3, using multipart uploading if applicable :param str…, uploadFile(), uploadFromPath()

### Community 186 - "conversions.py"
Cohesion: 0.07
Nodes (29): bytes2human(), bytes_in_unit(), convert_units(), hms_duration_to_seconds(), human2bytes(), mib_to_b(), parse_memory_string(), SupportsInt (+21 more)

### Community 187 - "replay_message_bus"
Cohesion: 0.29
Nodes (5): Replay all the messages and work out what they mean for jobs. We track the…, replay_message_bus(), Path, Make sure writing bus messages to files works with enums., Prints a list of the currently running jobs

### Community 188 - "safeFileTest.py"
Cohesion: 0.16
Nodes (10): Checkpointer, ABC, Base class for hooking checkpoints into operations., Install patches for this checkpointer. Each checkpointer provides its own…, Checkpointer that pauses during file read., Patch open to wrap read operations with checkpoint hooks., # TODO: Add tests for AtomicFileCreate path (concurrent new file creation)., # TODO: The 0.1s timeout waits to verify blocking are effectively sleeps; (+2 more)

### Community 189 - "LockCheckpointer"
Cohesion: 0.10
Nodes (11): LockCheckpointer, Acquire a shared lock (blocks if exclusive lock held)., Acquire an exclusive lock (blocks if any lock held)., Release whatever lock this thread holds., Checkpointer that pauses after flock acquisition., Associate a file descriptor with a path., Get the lock for a file descriptor., Get the lock for a path (for test assertions). (+3 more)

### Community 190 - "find_hanging_tests.py"
Cohesion: 0.24
Nodes (11): build_collect_from_pytest_args(), collect_expected_tests(), extract_pytest_commands_from_log(), fetch_url(), main(), parse_completed_tests(), Scan CI log output for test results and return the set of completed test IDs.…, Fetch a URL and return its decoded text content. (+3 more)

### Community 191 - "ReadableFileObj"
Cohesion: 0.29
Nodes (5): file_digest(), Protocol, Protocol that is more specific than what file_digest takes as an argument. Also…, Polyfilled hashlib.file_digest that works on Python <3.11., ReadableFileObj

### Community 192 - "realtimeLogger.py"
Cohesion: 0.29
Nodes (5): LoggingDatagramHandler, Receive logging messages from the jobs and display them on the leader. Uses…, Handle a single message. SocketServer takes care of splitting out the messages.…, # TODO: Protect the arguments better by actually pickling, # TODO: Format the message on the sending side?

### Community 193 - "find_default_container"
Cohesion: 0.33
Nodes (5): Builder, JobBase, find_default_container(), Hook the InitialWorkDirRequirement setup to make sure that there are no name…, Find the default constructor by consulting a Toil.options object.

### Community 194 - "Side-by-side monthly totals"
Cohesion: 0.33
Nodes (6): All models at a glance (100 GB, Scenario 1), Annual view (Scenario 1), Scale projection (live job store size), Scenario 1 — Single latest zip on S3 (common), Scenario 2 — 12 months zip retention, Side-by-side monthly totals

### Community 195 - "Cost models compared"
Cohesion: 0.33
Nodes (6): Cost models compared, Model A — Current (EBS primary + cron zip to S3), Model B — Prefix-based S3 primary (proposed Toil feature), Model C — Dedicated S3 bucket primary (supported today, no code change), Model D — Worst case (S3 primary but keep EBS), Model E — JuiceFS `file:` job store (supported today, no Toil code change)

### Community 196 - "TaskRunner"
Cohesion: 0.14
Nodes (9): cancel_run(), Send a signal to the process that is running Celery task task_id., Abstraction over the Celery API. Runs our run_wes task and allows canceling it.…, Cancel the task with the given ID on Celery., Returns True if the task has not yet failed, and False otherwise. Returns True…, Returns True if the task has not yet stopped, and False otherwise., TaskRunner, Return the state of the current run. Responsible for correcting the state when… (+1 more)

### Community 197 - "Recommendations"
Cohesion: 0.33
Nodes (6): Decision matrix, Hybrid (balance), If balancing cost, restart speed, and Slurm fit (recommended for this cluster), If fastest `--restart` is the priority, If lowest AWS cost is the priority, Recommendations

### Community 198 - "url_plugin_test.py"
Cohesion: 0.32
Nodes (3): FakeURLPlugin, IO, ParseResult

### Community 199 - "restartDAGTest.py"
Cohesion: 0.30
Nodes (8): failingFn(), passingFn(), Path, This function is guaranteed to pass as it does nothing out of the ordinary. If…, This function is guaranteed to fail via a raised assertion, or an os.kill…, Tests that restarted job DAGs don't run children of jobs that failed in the…, Creates a diamond DAG /->passingParent-\ root |-->child \\->failingParent--/…, TestRestartDAG

### Community 200 - ".addService"
Cohesion: 0.40
Nodes (3): Service, Add a service. The :func:`toil.job.Job.Service.start` method of the service…, Return True if the given Service is a service of this job, and False otherwise.

### Community 202 - ".__getstate__"
Cohesion: 0.33
Nodes (3): Return the dict to use as the instance's __dict__ when pickling., Return a semantically-shallow copy of the object, for :meth:`copy.copy`., Return a semantically-deep copy of the object, for :meth:`copy.deepcopy`.

### Community 203 - "MessageInbox"
Cohesion: 0.08
Nodes (18): MessageBusClient, MessageBusConnection, MessageInbox, Get a connection object that serves as an inbox for messages of the given…, Base class for clients (inboxes and outboxes) of a message bus. Handles keeping…, Make a disconnected client., Connect to the given bus. Should only ever be called once on a given instance., A buffered connection to a message bus that lets us receive messages. Buffers… (+10 more)

### Community 204 - "addOptions"
Cohesion: 0.22
Nodes (8): OptionParser, addOptions(), ensure_config(), ArgumentParser, If the config file at the filepath does not exist, create it. The parent…, Add all Toil command line options to a parser. Support for config files if…, ArgumentParser, Adds the default toil options to an :mod:`optparse` or :mod:`argparse` parser…

### Community 205 - ".url_exists"
Cohesion: 0.33
Nodes (3): Return True if the item at the given URL exists, and Flase otherwise. May raise…, Return True if the file at the given URI exists, and False otherwise. May raise…, HTTPServer

### Community 206 - "._setLeaderWorkerAuthentication"
Cohesion: 0.33
Nodes (3): Configure authentication between the leader and the workers. Assumes that we…, Generate a key pair, save it in /root/.ssh/id_rsa.pub on the leader, and return…, Get the Kubernetes joining info created when Kubernetes was set up on this…

### Community 207 - "scripts/tutorial_stats.py"
Cohesion: 0.47
Nodes (3): main(), think(), TimeWaster

### Community 208 - "google_retry"
Cohesion: 0.17
Nodes (4): google_retry(), Yields a context manager that can be used to read from the bucket with a…, This decorator retries the wrapped function if google throws any angry service…, compat_bytes()

### Community 209 - "connect_to_workflow_state_store"
Cohesion: 0.12
Nodes (12): connect_to_state_store(), connect_to_workflow_state_store(), Connect to a place to store state for workflows, defined by a URL. URL may be a…, Connect to a place to store state for the given workflow, in the state store…, Any, Run a requested workflow. :param base_scratch_dir: Directory where the…, Run the given task args with the given ID on Celery., Set up logging for the process into the given file and then call run_wes_task… (+4 more)

### Community 210 - "checksum.py"
Cohesion: 0.15
Nodes (9): get_s3_multipart_chunk_size(), Returns the chunk size of the S3 multipart object, given a file's size in bytes., compute_checksum_for_content(), compute_checksum_for_file(), Etag, BinaryIO, BytesIO, A hasher for s3 etags. (+1 more)

### Community 211 - "slurm_recovery_workflow.py"
Cohesion: 0.47
Nodes (5): main(), Dispatch a single child that simulates one storage failure then succeeds on…, Fail once with a storage-like error, then return on retry. Uses a marker file…, setup_storage_failure_test(), storage_failure_child()

### Community 212 - "examples/example_cachingbenchmark.py"
Cohesion: 0.70
Nodes (4): main(), poll(), report(), root()

### Community 213 - ".open"
Cohesion: 0.29
Nodes (3): Create the context manager around tasks prior and after a job has been run.…, Log a report of the files accessed. Includes the files that were accessed while…, Send a logging message to the leader. The message will also be \ logged by the…

### Community 214 - "._runOrphanedDeferredFunctions"
Cohesion: 0.22
Nodes (5): Yields a single-argument function that allows for deferred functions of type…, Run a deferred function (either our own or someone else's). Reports an error if…, Read and run deferred functions until EOF from the given open file., Run all of the deferred functions that were registered., Scan for files that aren't locked by anybody and run all their deferred…

### Community 215 - "ROADMAP.md"
Cohesion: 0.40
Nodes (4): Completed, Longer Term, Medium Term (~ 6-month goals, by ~June 2018?), Near Term (In Progress, Estimated Completion Date?)

### Community 216 - "get_object_for_url"
Cohesion: 0.09
Nodes (13): parse_jobstore_identifier(), ParseResult, Do we need both get_file_size and _get_size???, Do we need both get_file_size and _get_size???, establish_boto3_session(), Get a Boto 3 session usable by the current thread. This function may not always…, get_object_for_url(), list_objects_for_url() (+5 more)

### Community 217 - ".wrapFn"
Cohesion: 0.40
Nodes (3): Makes a Job out of a function. Convenience function for constructor of…, noOp(), Make sure that the encapsulate child does not have two parents with unique…

### Community 218 - "helloWorldTest.py"
Cohesion: 0.27
Nodes (4): childFn(), FollowOn, HelloWorld, HelloWorldTest

### Community 219 - "fn1Test"
Cohesion: 0.38
Nodes (6): encapsulatedJobFn(), Path, Tests the Job.encapsulation method, which uses the EncapsulationJob class., fn1Test(), FileDescriptorOrPath, Function appends the next character after the last character in the given…

### Community 220 - "realtimeLoggerTest.py"
Cohesion: 0.27
Nodes (4): LogTest, MessageDetector, Detect the secret message and set a flag., RealtimeLoggerTest

### Community 221 - "helloWorld.py"
Cohesion: 0.19
Nodes (6): hello_world(), hello_world_child(), main(), # NOTE: path and the udpated file are stored to /tmp, Path, RegularLogTest

### Community 222 - ".run"
Cohesion: 0.40
Nodes (3): Any, Run the leader process to issue and manage jobs. :raises:…, Create a file in the jobstore indicating failure or success.

### Community 223 - "scripts/example_cachingbenchmark.py"
Cohesion: 0.70
Nodes (4): main(), poll(), report(), root()

### Community 225 - "Checkpoint"
Cohesion: 0.13
Nodes (8): Checkpoint, Register a checkpoint for the given thread., Get checkpoint for thread, if any., A synchronization point that allows a test to pause a thread and know when the…, Signal that we've arrived at the checkpoint, then wait for release. Returns…, Wait for a thread to arrive at this checkpoint. Returns True if arrived, False…, Release the thread waiting at this checkpoint., Check if a thread has arrived (non-blocking).

### Community 226 - "AWS storage pricing reference (approximate)"
Cohesion: 0.50
Nodes (4): Annual storage-only comparison (100 GB), AWS storage pricing reference (approximate), Per GB-month (100 GB extrapolation), S3 API requests (same region, via Gateway VPC endpoint)

### Community 227 - "add_paths"
Cohesion: 0.29
Nodes (6): add_paths(), all_parents(), Yield all parents of the given path, up to the filesystem root. All yielded…, Based off of WDL.runtime.task_container.add_paths from miniwdl Comes up with a…, Inject extra Bash code from the Toil WDL runtime into the command for the…, TaskContainer

### Community 228 - "._try_terminate"
Cohesion: 0.25
Nodes (3): Internal function. Should not be called outside this class. Try to terminate an…, Internal function. Should not be called outside this class. Wait for a…, Internal function. Should not be called outside this class. Destroy any job…

### Community 229 - "Current Toil AWS job store behavior"
Cohesion: 0.50
Nodes (4): Current Toil AWS job store behavior, Locator format, Prefix-based job store (proposed, not implemented), What the implementation does

### Community 230 - ".scan_bus_messages"
Cohesion: 0.11
Nodes (15): Listener, MessageType, bytes_to_message(), Any, FileDescriptorOrPath, IO, Convert bytes from message_to_bytes back to a message of the given type., Convert a type to a name that can be a PyPubSub topic (all normal characters,… (+7 more)

### Community 231 - "Non-cost tradeoffs"
Cohesion: 0.50
Nodes (4): Non-cost tradeoffs, Operational, Prefix vs dedicated bucket (cost-neutral), Restart latency

### Community 234 - "PULL_REQUEST_TEMPLATE.md"
Cohesion: 0.50
Nodes (3): Changelog Entry, Merger Checklist, Reviewer Checklist

### Community 238 - "IAMTest"
Cohesion: 0.22
Nodes (3): mock_aws, IAMTest, Check that given permissions and associated functions perform correctly

### Community 239 - ".test_writer_paused_mid_write_blocks_reader"
Cohesion: 0.33
Nodes (4): Checkpointer that pauses during file write., Patch open to wrap write operations with checkpoint hooks., Verify that a reader is blocked even when writer is paused during the actual…, WriteCheckpointer

### Community 241 - "get_requirements"
Cohesion: 0.38
Nodes (6): get_requirements(), import_version(), Return the module object for src/toil/version.py, generate from the template if…, Load the requirements for the given extra. Uses the appropriate requirements-…, Call setup(). This function exists so the setup() invocation preceded more…, run_setup()

### Community 242 - ".getDefaultOptions"
Cohesion: 0.50
Nodes (3): Namespace, StrPath, Get default options for a toil workflow. :param jobStore: A string describing…

### Community 244 - ".__init__"
Cohesion: 0.50
Nodes (3): Any, IO, Wrap the given backing stream.

### Community 245 - "ImportWorkersMessageHandler"
Cohesion: 0.33
Nodes (4): ImportWorkersMessageHandler, LogRecord, Detect whether any WorkerImportJob jobs ran during a workflow., _stream_handler

### Community 246 - "Runner"
Cohesion: 0.33
Nodes (3): make_parser(), ArgumentParser, Runner

### Community 248 - "safe_write_file"
Cohesion: 0.50
Nodes (3): Safely write to a file by acquiring an exclusive lock to prevent other…, Set or clear a key value on the filesystem., safe_write_file()

### Community 251 - "strip_trailing_whitespace_from_all_files_in_dir"
Cohesion: 0.47
Nodes (5): main(), Strips trailing whitespace from a file, in-place., Strips trailing whitespace from all files in a directory, recursively. Only…, strip_trailing_whitespace_from_all_files_in_dir(), strip_trailing_whitespace_from_file()

### Community 253 - "TimeWaster"
Cohesion: 0.47
Nodes (3): main(), think(), TimeWaster

### Community 254 - "toil/conftest.py"
Cohesion: 0.33
Nodes (5): Item, pytest_collection_modifyitems(), # TODO: Pytest doesn't expose the types we need to use to type this publicly, # TODO: Pytest also doesn't expose the class we need to sniff for., Apply a timeout to all test items that extend _pytest.doctest.DoctestItem.

### Community 258 - ".get_toil_coordination_dir"
Cohesion: 0.29
Nodes (4): Return a path to a writable directory, which will be in memory if convenient.…, Get a safe filesystem path component for a workflow. Will be consistent for all…, Try to use the given path. Return it if it exists or can be made, and we can…, try_path()

### Community 260 - "FileResource"
Cohesion: 0.33
Nodes (3): FileResource, BinaryIO, A resource read from a file on the leader.

### Community 261 - ".private_history_manager"
Cohesion: 0.50
Nodes (3): fixture, MonkeyPatch, Path

### Community 266 - ".readGlobalFileStream"
Cohesion: 0.40
Nodes (3): IO, Read a stream from the job store; similar to readGlobalFile. The yielded file…, Send a stream of UTF-8 text to the leader as a named log stream. Useful for…

### Community 267 - "applianceSelf"
Cohesion: 0.21
Nodes (11): applianceSelf(), Return the fully qualified name of the Docker image to start Toil appliance…, opt_strtobool(), Convert an optional string representation of bool to None or bool, check_valid_node_types(), parse_node_types(), Parse a specification for zero or more node types. Takes a comma-separated list…, Raises if an invalid nodeType is specified for aws or gce. :param str… (+3 more)

### Community 269 - "enable_absolute_imports"
Cohesion: 0.67
Nodes (3): enable_absolute_imports(), main(), Empty modules >>> enable_absolute_imports('') 'from __future__ import…

### Community 271 - "wdltoil.py"
Cohesion: 0.02
Nodes (95): Document, ReadSourceResult, potential_absolute_uris(), Get potential absolute URIs to check for an imported file. Given a URI or bare…, convert_remote_files(), main(), # TODO: For WDL 1.2, this needs to handle directories and also recursively, # TODO: Implement directory virtualization here! (+87 more)

### Community 272 - "._testJobFileStore"
Cohesion: 0.14
Nodes (11): explode(), main(), This workflow always fails.      Invoke like:          python examples/example_a, analysisJob(), parentJob(), stageFn(), fileTestJob(), Creates a chain of jobs, each reading and writing files using the… (+3 more)

### Community 274 - "rootpath"
Cohesion: 0.40
Nodes (4): FixtureRequest, fixture, Records the rootpath at the class level, for use on a unittest.TestCase., rootpath()

### Community 275 - ".add_options"
Cohesion: 0.50
Nodes (3): _ArgumentGroup, ArgumentParser, If this batch system provides any command line options, add them to the given…

### Community 295 - "atomic_copyobj"
Cohesion: 0.40
Nodes (4): Copy a file from the file store into the cache. Will hardlink if appropriate.…, atomic_copyobj(), BytesIO, Copy an open file using posix atomic creations semantics.

## Knowledge Gaps
- **70 isolated node(s):** `podKiller.sh script`, `run.sh script`, `customDockerInit.sh script`, `singularity-wrapper.sh script`, `waitForKey.sh script` (+65 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **49 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Toil` connect `Job` to `JobDescription`, `InsufficientSystemResources`, `Info`, `toil/common.py`, `.get_toil_coordination_dir`, `sortTest.py`, `AbstractFileStore`, `AbstractProvisioner`, `Leader`, `toil/test/__init__.py`, `Shape`, `TestImportExportFile`, `toil-sort-example.py`, `AWSJobStore`, `._testJobFileStore`, `wdltoil.py`, `toilContextManagerTest.py`, `BatchJobExitReason`, `GoogleJobStore`, `HistoryManager`, `RealtimeLogger`, `integrative`, `statsAndLogging.py`, `AutoDeploymentTest`, `create_current_submission`, `main`, `.wrapJobFn`, `ToilMetrics`, `utilsTest.py`, `awsBatch.py`, `slow`, `fileStoreTest.py`, `AbstractCachingFileStoreTest`, `TestResource`, `strtobool`, `FileID`, `environmentTest.py`, `TestUtils`, `.__init__`, `serverTest.py`, `tasks.py`, `.testConcurrencyWithDisk`, `toilStats.py`, `TaskRunner`, `abstractBatchSystem.py`, `restartDAGTest.py`, `decode_directory`, `scripts/tutorial_stats.py`, `slurm_recovery_workflow.py`, `examples/example_cachingbenchmark.py`, `FileJobStore`, `helloWorld.py`, `scripts/example_cachingbenchmark.py`, `threading.py`, `Submission`, `Config`, `AbstractJobStore`, `HelloWorld`, `HelloWorld`, `ToilRestartException`, `makeRootJob`, `apiDockerCall`, `history_submission.py`, `options/common.py`, `SortTest`, `TestJob`, `worker.py`, `slurmSortTest.py`, `TimeWaster`?**
  _High betweenness centrality (0.143) - this node is a cross-community bridge._
- **Why does `Job` connect `Job` to `UserDefinedJobArgTypeTest`, `JobDescription`, `InsufficientSystemResources`, `AbstractFileStore`, `toil/common.py`, `sortTest.py`, `CachingFileStore`, `stress.py`, `toil/test/__init__.py`, `jobServiceTest.py`, `TestImportExportFile`, `wdltoil.py`, `setter`, `toilContextManagerTest.py`, `BatchJobExitReason`, `integrative`, `AutoDeploymentTest`, `TestCleanWorkDir`, `.wrapJobFn`, `ToilTest`, `ToilMetrics`, `utilsTest.py`, `slow`, `fileStoreTest.py`, `resumabilityTest.py`, `AbstractCachingFileStoreTest`, `TestResource`, `FileID`, `._loadUserModule`, `promisedRequirementTest.py`, `deferredFunctionTest.py`, `environmentTest.py`, `TestUtils`, `Any`, `.__init__`, `TestJobService`, `Any`, `.testConcurrencyWithDisk`, `toilStats.py`, `restartDAGTest.py`, `.addService`, `.addChild`, `scripts/tutorial_stats.py`, `slurm_recovery_workflow.py`, `.open`, `RuntimeError`, `FileJobStore`, `.wrapFn`, `helloWorldTest.py`, `._parseResource`, `helloWorld.py`, `fn1Test`, `scripts/example_cachingbenchmark.py`, `realtimeLoggerTest.py`, `Config`, `AbstractJobStore`, `HelloWorld`, `HelloWorld`, `ToilRestartException`, `apiDockerCall`, `slurmSortTest.py`, `SortTest`, `TestJob`, `worker.py`, `.run`, `TimeWaster`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **Why does `Config` connect `Config` to `Job`, `JobDescription`, `InsufficientSystemResources`, `Info`, `toil/common.py`, `AbstractFileStore`, `MockBatchSystemAndProvisioner`, `AbstractProvisioner`, `Test`, `Leader`, `Shape`, `SlurmTest`, `AWSJobStore`, `GoogleJobStoreTest`, `BatchJobExitReason`, `GoogleJobStore`, `HistoryManager`, `RealtimeLogger`, `.test_symlink_read_control`, `statsAndLogging.py`, `iam.py`, `slurm.py`, `GridEngineThread`, `main`, `ToilMetrics`, `utilsTest.py`, `awsBatch.py`, `retry`, `strtobool`, `GridEngineThread`, `TestUtils`, `MesosBatchSystem`, `toilStats.py`, `abstractBatchSystem.py`, `TestSlurmMountRecovery`, `FileJobStore`, `toil/__init__.py`, `.session`, `threading.py`, `AbstractJobStore`, `ToilRestartException`, `leader.py`, `worker.py`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Are the 179 inferred relationships involving `Job` (e.g. with `HelloWorld` and `HelloWorld`) actually correct?**
  _`Job` has 179 INFERRED edges - model-reasoned connections that need verification._
- **Are the 223 inferred relationships involving `Toil` (e.g. with `main()` and `main()`) actually correct?**
  _`Toil` has 223 INFERRED edges - model-reasoned connections that need verification._
- **Are the 116 inferred relationships involving `JobDescription` (e.g. with `AbstractBatchSystem` and `AbstractScalableBatchSystem`) actually correct?**
  _`JobDescription` has 116 INFERRED edges - model-reasoned connections that need verification._
- **Are the 175 inferred relationships involving `Config` (e.g. with `AbstractBatchSystem` and `AbstractScalableBatchSystem`) actually correct?**
  _`Config` has 175 INFERRED edges - model-reasoned connections that need verification._