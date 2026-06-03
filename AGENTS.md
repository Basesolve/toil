## Learned User Preferences

- Slurm validation runs on a remote AWS ParallelCluster head node only; do not assume local Slurm is available.
- When Slurm behavior must be checked outside pytest, give direct `python` entry-point commands with required env vars and paths.
- Slurm end-to-end tests should follow `sortTest.py` patterns: `@integrative`, `@needs_slurm`, `ToilTest` subclass, and `toil clean` in teardown.
- Deploy patched Toil to the remote venv by copying `slurm.py` into site-packages or running `pip install -e .` from the repo root.
- For remote-only Slurm bugs, provide manual verification steps (grep installed module, static init-order checks, re-run workflow) rather than only local pytest.
- Run Python tooling with `.venv/bin/python` first, falling back to `./venv/bin/python` when needed.

## Learned Workspace Facts

- Slurm mount-recovery logic lives in `src/toil/batchSystems/slurm_mount_recovery.py` and `src/toil/batchSystems/slurm.py`.
- `SlurmBatchSystem` must set `partition_switch_watch`, `_lost_job_first_seen`, and `_partition_switch_last_poll` before `super().__init__()` because the grid-engine thread starts there.
- Job status details use `JobStatusDetail` 3-tuples `(state, exit_code, reason)`; `_getJobDetailsFromScontrol` and `_getJobDetailsFromSacct` must match.
- `submitJob` returns string Slurm IDs; use `slurm_job_number()` when reading `batchJobIDs` entries that may be int or str.
- Slurm integration tests: `src/toil/test/sort/slurmSortTest.py` and `src/toil/test/sort/slurm_recovery_workflow.py`.
- Cluster integration tests require `TOIL_TEST_INTEGRATIVE=True`, `TOIL_TEST_SLURM_SHARED_DIR`, `TOIL_TEST_SLURM_COORD_DIR`, and `TOIL_SLURM_PARTITION`.
- Slurm unit tests live in `src/toil/test/batchSystems/test_slurm.py` (`SlurmTest`, `TestSlurmMountRecovery`).
- `parse_slurm_nodelist` must split commas only at bracket depth zero (for example `cn[001-003,005]`).
- `batch_logs_indicate_storage_failure` scans only the current Slurm attempt's stdout/stderr logs, not all retries for a Toil job.
- Remote deployment context uses paths like `/opt/augmet/augmet-engine-ro` and shared job stores under `/augmet-mp/job_stores/`.
