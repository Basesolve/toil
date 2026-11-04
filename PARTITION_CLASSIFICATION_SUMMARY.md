# Slurm Partition Classification & Unified Partition-Switch Management

This document describes the new spot / on-demand partition classification and the
unified partition-switch mechanism, and provides cutover notes for older
deployments.

## What changed

Previously, Toil could not tell whether a Slurm partition was **spot** or
**on-demand**. Spot-ness was inferred from a single name pattern
(`TOIL_SLURM_PARTITION_PREFERED`), and the partition-switch (storage-failover)
logic was a separate, hard-coded path. This made spot failovers fragile and the
two mechanisms impossible to coordinate.

The change has two parts:

1. **Explicit partition classification** — a three-level precedence for deciding
   whether each partition is `spot`, `on_demand`, or `unknown`.
2. **Unified partition-switch management** — a cached `PartitionRegistry` plus a
   single, precedence-ordered placement decision used both for normal placement
   and for failovers.

---

## 1. Partition classification

### Precedence (highest wins)

For each partition, classification is decided by the first rule that yields a
non-`unknown` result:

1. **Explicit globs** — `--slurmSpotPartitions` / `TOIL_SLURM_SPOT_PARTITIONS`
   and `--slurmOnDemandPartitions` / `TOIL_SLURM_ONDEMAND_PARTITIONS`.
   Comma-separated glob patterns matched against the partition name
   (e.g. `cpu-spot-*,gpu-spot`).
2. **Slurm node Features** — if no explicit glob matched, the partition's nodes'
   `Features` (from `sinfo -o %f`) are searched for the spot-feature tag
   (`--slurmSpotFeature` / `TOIL_SLURM_SPOT_FEATURE`, default `spot`). If any
   node feature contains that tag, the partition is `spot`.
3. **Legacy name pattern** — if neither of the above classified **any**
   partition, the legacy `TOIL_SLURM_PARTITION_PREFERED` name pattern is applied
   as before.

If **nothing** classifies **any** partition, **no `preference` column is added**
to the resources DataFrame — behavior is byte-identical to the legacy code for
unclassified clusters.

### Why `sinfo %f` (node Features) and not `scontrol show partition`

`PartitionName` / `Alternate=` / `State` come from `scontrol`, but **Slurm does
not expose a spot/on-demand flag on partitions**. The only operator-controlled
per-node metadata available without admin privileges is the node `Features`
string. AWS ParallelCluster and many site configs already tag spot nodes with a
feature (commonly `spot`), so `sinfo -o %f` is the most portable, least
intrusive source of truth. It is read with
`sinfo -a -o "%P %G %l %p %c %m %f"` and is tolerant of older Slurm builds that
lack `%f` (treated as empty) and of `(null)` feature lists.

---

## 2. Unified partition-switch management

### PartitionRegistry

A new cached `PartitionRegistry` (per `SlurmBatchSystem` instance) holds each
partition's `State`, `Alternate=`, classification, and memory capacity. It
refreshes from a **single batched** `scontrol show partition -o` call when the
cache is older than `--slurmPartitionCacheTTL` / `TOIL_SLURM_PARTITION_CACHE_TTL`
(default **60 s**).

This replaces the previous per-job
`scontrol show partition ... | sed | grep | cut` pipelines.

### Placement precedence

When a job is submitted, the target partition is chosen by the first rule that
applies (`resolve_partition_for_job`):

1. **Per-job spot failover** (opt-in, `--slurmSpotFailover`) — after N strikes
   the job is retried onto the failed partition's `Alternate=` (or failover-map /
   failover-list fallback).
2. **Global storage failover** — if the mount-recovery logic flagged a storage
   failure, place on the configured failover partition.
3. **`select_partition`** — normal placement, now using the registry's cached
   `State` instead of shell pipelines, and **no longer recursing unboundedly**
   (it tries the preferred class, then the other class, then returns `None`).

Both failover kinds skip partitions whose memory capacity is too small for the
job.

### Failover-target resolution

`resolve_spot_failover_partition` picks the failover target in this order:

1. **`--slurmPartitionFailoverMap`** / `TOIL_SLURM_PARTITION_FAILOVER_MAP`
   (format `spot-a:od-a,spot-b:od-b`) — explicit spot→on-demand mapping.
2. **Slurm `Alternate=`** — the partition's own `Alternate=` attribute.
3. **`--slurmPartitionFailover`** / `TOIL_SLURM_PARTITION_FAILOVER` — the
   legacy failover list (unchanged).

### Single-node / single-partition safety

With a single partition and no `Alternate=`, every rule degrades to a safe
no-op: no failover partition is selected, a warning is logged, a self-map is
harmless, and `select_partition` returns `None` instead of raising
`RecursionError`.

---

## 3. New / promoted options

All are configargparse options backed by environment variables; CLI takes
precedence over env. Defaults are `None` (unset) unless noted.

| CLI option | Env var | Purpose |
|---|---|---|
| `--slurmSpotPartitions` | `TOIL_SLURM_SPOT_PARTITIONS` | Comma-separated globs for spot partitions |
| `--slurmOnDemandPartitions` | `TOIL_SLURM_ONDEMAND_PARTITIONS` | Comma-separated globs for on-demand partitions |
| `--slurmSpotFeature` | `TOIL_SLURM_SPOT_FEATURE` | Node-feature tag marking spot (default `spot`) |
| `--slurmPartitionFailoverMap` | `TOIL_SLURM_PARTITION_FAILOVER_MAP` | `spot-a:od-a,...` spot→on-demand map |
| `--slurmPartitionCacheTTL` | `TOIL_SLURM_PARTITION_CACHE_TTL` | Registry cache TTL seconds (default 60) |
| `--slurmJobRestartThreshold` | `TOIL_SLURM_JOB_RESTART_THRESHOLD` | Promoted from env-only |
| `--slurmPartitionSwitchCooldown` | `TOIL_SLURM_PARTITION_SWITCH_COOLDOWN` | Promoted from env-only |
| `--slurmPartitionSwitchPollInterval` | `TOIL_SLURM_PARTITION_SWITCH_POLL_INTERVAL` | Promoted from env-only |

---

## 4. Cutover notes for older deployments

The change is **backward compatible**: if you set none of the new options,
behavior is identical to before.

### If you use `TOIL_SLURM_PARTITION_PREFERED` (name pattern)

- **No action required.** The legacy pattern still works as the lowest-priority
  rule.
- **One subtlety fixed:** previously, if your nodes carried *any* Slurm
  `Features` (e.g. ParallelCluster's `m6i.large,dynamic`) the legacy pattern
  could be silently disabled. The gate is now "only when nothing else classified
  any partition", so the legacy pattern works again on untagged-but-featured
  clusters.

### To adopt explicit classification (recommended)

Pick **one** of:

- **Globs (no Slurm changes needed):**
  ```bash
  export TOIL_SLURM_SPOT_PARTITIONS="cpu-spot-*,gpu-spot"
  export TOIL_SLURM_ONDEMAND_PARTITIONS="cpu-od,gpu-od"
  ```
- **Node features (if your nodes are tagged):** ensure spot nodes carry a
  feature such as `spot`, then optionally override the tag:
  ```bash
  export TOIL_SLURM_SPOT_FEATURE=spot   # default
  ```

Explicit globs / features take precedence over the legacy pattern; once any
partition is classified, the legacy pattern is not consulted.

### To adopt the failover map (replacing `Alternate=` / failover list)

```bash
export TOIL_SLURM_PARTITION_FAILOVER_MAP="cpu-spot:cpu-od,gpu-spot:gpu-od"
```
This takes precedence over `Alternate=` and over `--slurmPartitionFailover`.
If unset, the existing `Alternate=` / failover-list behavior is unchanged.

### Spot failover deployments (`TOIL_SLURM_SPOT_FAILOVER=1`)

No change required. The failover-target lookup now consults the map first, then
`Alternate=`, then the list — so existing `Alternate=`-based ParallelCluster
setups (`PartitionName=*-cs Alternate=*-cod`) keep working.

---

## 5. Files changed

- `src/toil/batchSystems/slurm_mount_recovery.py` — pure helpers
  (`parse_glob_list`, `classify_partition`), `PartitionClass`,
  `DEFAULT_SPOT_FEATURE`, `DEFAULT_PARTITION_CACHE_TTL`.
- `src/toil/batchSystems/slurm.py` — `PartitionInfo.features`,
  `PartitionRegistry`, reworked `select_partition` / `resolve_partition_for_job`
  / `get_alternate_partition` / `resolve_spot_failover_partition`, classification
  plumbing in `assessBatchResources`, and the 8 new options.
- `src/toil/test/batchSystems/test_slurm.py` — 20 new tests.
- `AGENTS.md` — workspace-facts update.

## 6. Testing

Unit tests: `src/toil/test/batchSystems/test_slurm.py`
(`83 passed, 11 failed`; the 11 failures are pre-existing on clean HEAD and
unrelated). No local Slurm is available — end-to-end validation must run on the
remote AWS ParallelCluster head node with `TOIL_TEST_INTEGRATIVE=True` and the
Slurm env vars set.
