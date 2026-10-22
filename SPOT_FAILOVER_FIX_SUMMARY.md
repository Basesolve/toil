# SLURM Spot Failover with doubleMem Fix

## Problem
When SLURM spot failover is enabled and doubleMem is triggered, jobs could fail with "Memory specification can not be satisfied" because the failover partition precedence logic forced jobs to use specific partitions even when those partitions couldn't accommodate the doubled memory requirements.

## Root Cause
The partition selection logic in `src/toil/batchSystems/slurm.py` gave highest precedence to per-job spot failover partitions, bypassing the normal memory-based partition selection entirely. Once a failover partition was set for a job, ALL subsequent attempts used that partition.

## Solution
Modified the partition selection logic to check if failover partitions can handle the memory requirements before using them. When a failover partition cannot accommodate the memory requirements, the logic falls back to normal memory-based partition selection.

## Changes Made

### 1. Added Partition Memory Capacity Detection Method
**File**: `src/toil/batchSystems/slurm.py` (lines 357-385)

Added `get_partition_memory_capacity()` method to the `PartitionSet` class that:
- Parses partition memory strings (e.g., "32000", "64G", "128M") 
- Returns memory capacity in megabytes
- Handles various memory unit formats
- Returns `None` for unknown partitions or parsing errors

### 2. Enhanced Partition Selection Logic
**File**: `src/toil/batchSystems/slurm.py` (lines 1674-1705)

Modified the partition precedence logic to:
1. **Check per-job failover partitions**: Before using a per-job failover partition, verify it can handle the current memory requirements
2. **Check global failover partitions**: Before using a global failover partition, verify it can handle the memory requirements  
3. **Fall back to normal selection**: When failover partitions cannot handle memory requirements, allow normal memory-based partition selection to proceed

### 3. Added Informative Logging
Added detailed log messages that indicate:
- When a failover partition is skipped due to memory constraints
- The memory requirement vs partition capacity details
- When normal partition selection is triggered

## Testing

The implementation was verified through:
1. **Syntax validation**: Code compiles without errors
2. **Memory parsing tests**: Verified correct parsing of various memory string formats
3. **Logic validation**: Tested the decision logic for different memory scenarios
4. **Scenario testing**: Validated the end-to-end spot failover with doubleMem scenario

## Expected Behavior

### Before Fix
```
1. Job fails on spot partition → moves to failover partition
2. Job fails on failover partition → doubleMem doubles memory  
3. Job resubmitted to same failover partition → "Memory specification can not be satisfied"
```

### After Fix
```
1. Job fails on spot partition → moves to failover partition
2. Job fails on failover partition → doubleMem doubles memory
3. System checks if failover partition can handle doubled memory
4. If yes: Use failover partition
5. If no: Use normal memory-based partition selection → success
```

## Files Modified
- `src/toil/batchSystems/slurm.py`: Added method and modified partition selection logic

## Impact
- ✅ Maintains existing spot failover functionality
- ✅ Preserves doubleMem behavior  
- ✅ Prevents job failures due to inadequate failover partition resources
- ✅ Provides better diagnostics through improved logging
- ✅ No breaking changes to existing APIs or configurations