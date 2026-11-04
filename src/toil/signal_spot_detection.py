# Copyright (C) 2015-2021 Regents of the University of California
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
# See the License for the specific language governing permissions

"""
Signal-based spot interruption detection for Toil workers.

This module provides mechanisms for workers to explicitly report when they
receive a catchable signal (default SIGTERM), enabling more accurate spot
interruption detection than exit code inference alone.

Note: SIGKILL cannot be caught, so a worker killed by SIGKILL (e.g. the OOM
killer or a hard ``scontrol`` kill) leaves no marker. Spot detection for those
deaths relies on exit-code / state inference on the batch-system side.
"""

import hashlib
import logging
import os
import tempfile
import time
from typing import Any

logger = logging.getLogger(__name__)

# Signal number received by this worker (set in handle_spot_signal)
_SPOT_SIGNAL_RECEIVED: int | None = None

# Module-level config, set by initialize_spot_signal_handling()
_JOB_STORE_ID: str = "unknown"
_MARKER_DIR: str | None = None


def _get_workflow_id() -> str:
    """
    Best-effort ID of the workflow this process belongs to.

    The leader exports ``TOIL_WORKFLOW_ID`` into each worker's environment
    (see Leader.issueJob), so a worker and its leader agree on it. On the
    leader itself (which sets the var only in the worker env, not its own),
    and in tests, fall back to the coordination dir. Two workflows sharing a
    marker dir and the same jobStoreID must not clobber each other's markers.
    """
    return (
        os.getenv("TOIL_WORKFLOW_ID")
        or os.getenv("TOIL_COORDINATION_DIR")
        or "unknown"
    )


def _get_spot_signal_signum() -> int:
    """
    Get the signal number to watch for spot interruption detection.

    Configurable via TOIL_SPOT_SIGNAL_SIG environment variable.
    Defaults to signal.SIGTERM.
    """
    import signal

    signal_name = os.getenv("TOIL_SPOT_SIGNAL_SIG", "SIGTERM")
    try:
        signum = int(getattr(signal, signal_name))
    except (AttributeError, ValueError):
        logger.warning(
            "Invalid signal name '%s' in TOIL_SPOT_SIGNAL_SIG, defaulting to SIGTERM",
            signal_name
        )
        return signal.SIGTERM
    if signum in (signal.SIGKILL, signal.SIGSTOP):
        logger.warning(
            "TOIL_SPOT_SIGNAL_SIG=%s cannot be caught; defaulting to SIGTERM",
            signal_name
        )
        return signal.SIGTERM
    return signum


def _resolve_marker_dir() -> str:
    """
    Resolve the directory used for spot signal marker files.

    Precedence:
      1. TOIL_SPOT_SIGNAL_DIR environment variable
      2. <TOIL_COORDINATION_DIR>/spot_signals (shared filesystem on grids)
      3. <tempdir>/toil_spot_signals (local fallback; NOT shared on grids)
    """
    explicit = os.getenv("TOIL_SPOT_SIGNAL_DIR")
    if explicit:
        return explicit
    coordination_dir = os.getenv("TOIL_COORDINATION_DIR")
    if coordination_dir:
        return os.path.join(coordination_dir, "spot_signals")
    fallback = os.path.join(tempfile.gettempdir(), "toil_spot_signals")
    logger.debug(
        "Spot signal marker dir falling back to node-local %s; "
        "markers will not be visible across nodes",
        fallback,
    )
    return fallback


def _marker_path(job_store_id: str, marker_dir: str | None = None) -> str:
    """
    Compute the marker file path for a Toil job.

    Keyed by jobStoreID only (not PID): at most one worker runs per Toil
    job at any time, and batch systems like Slurm never know the worker
    PID, so PID-keyed markers would be unreadable by the reader side.
    """
    directory = marker_dir or _MARKER_DIR or _resolve_marker_dir()
    # Namespace by workflow so two workflows that reuse a jobStoreID (or run
    # concurrently in the same marker dir) cannot clobber each other.
    digest = hashlib.md5(
        f"{_get_workflow_id()}:{job_store_id}".encode()
    ).hexdigest()[:16]
    return os.path.join(directory, f"spot_signal_{digest}.txt")


def initialize_spot_signal_handling(job_store_id: str | None = None) -> None:
    """
    Initialize spot signal handling for this worker.

    Sets the marker directory and this job's marker key, clears any stale
    marker left by a previous attempt of the same job (retry contamination),
    and returns the signal number the caller should register.

    :param job_store_id: This worker's Toil jobStoreID. Falls back to the
        TOIL_JOB_STORE_ID environment variable, then "unknown".
    :return: The signal number to register handle_spot_signal for.
    """
    global _JOB_STORE_ID, _MARKER_DIR

    _JOB_STORE_ID = job_store_id or os.getenv("TOIL_JOB_STORE_ID", "unknown")
    _MARKER_DIR = _resolve_marker_dir()

    try:
        os.makedirs(_MARKER_DIR, exist_ok=True)
    except OSError as e:
        logger.debug("Could not create spot signal directory %s: %s", _MARKER_DIR, e)

    # Clear any stale marker from a previous attempt of this job
    clear_spot_signal(_JOB_STORE_ID)

    signum = _get_spot_signal_signum()
    logger.debug(
        "Initialized spot signal handling: signal %d, marker %s",
        signum,
        _marker_path(_JOB_STORE_ID),
    )
    return signum


def handle_spot_signal(signal_number: int, frame: Any) -> None:
    """
    Handle SIGTERM as a spot interruption signal.
    
    This allows the worker to explicitly signal that it was terminated
    by the batch scheduler (Slurm) due to spot/preempt constraints,
    helping distinguish spot interruptions from normal application failures.
    
    :param signal_number: The signal number received
    :param frame: Current stack frame
    """
    global _SPOT_SIGNAL_RECEIVED
    _SPOT_SIGNAL_RECEIVED = signal_number
    
    import signal
    
    # Log the signal receipt
    try:
        signal_name = signal.Signals(signal_number).name
    except (AttributeError, ValueError):
        signal_name = str(signal_number)
    
    logger.warning(
        "Worker received spot/SIGTERM signal: %s. "
        "This may indicate a spot interruption or partition switch.",
        signal_name
    )
    
    # Write the marker file NOW, inside the handler, because the process
    # may be killed (SIGKILL) or exit before reaching cleanup code in main().
    try:
        report_spot_signal(os.getpid())
    except Exception as e:
        logger.debug("Could not write spot signal marker in handler: %s", e)

    # Restore default disposition and re-raise the signal so the worker
    # dies exactly as it would have without this handler (preserving
    # waitid()/waitpid() signal-death semantics for batch systems).
    # Without this, the worker would linger until SIGKILL and batch
    # systems would observe SIGKILL death instead of SIGTERM.
    try:
        signal.signal(signal_number, signal.SIG_DFL)
        os.kill(os.getpid(), signal_number)
    finally:
        # If re-raising somehow did not kill us (signal blocked/masked),
        # exit with the conventional 128+signum status so the death is
        # still reported as a signal-style termination to waitpid callers.
        os._exit(128 + signal_number)


def report_spot_signal(pid: int, job_store_id: str | None = None) -> str | None:
    """
    Report signal receipt to the batch system via an atomic marker file.

    The marker is keyed by jobStoreID (see _marker_path) so batch systems
    that never know the worker PID (e.g. Slurm) can still find it.
    PID, SLURM_JOB_ID, and timestamp are recorded in the file content
    for debugging.

    :param pid: Process ID of the worker (diagnostic only)
    :param job_store_id: Toil jobStoreID; defaults to the initialized value
    :return: Path to marker file on success, None on failure
    """
    if _SPOT_SIGNAL_RECEIVED is None:
        logger.debug("No spot signal received, not creating marker file")
        return None

    job_store_id = job_store_id or _JOB_STORE_ID
    marker_path = _marker_path(job_store_id)

    content = (
        f"signal={_SPOT_SIGNAL_RECEIVED}\n"
        f"pid={pid}\n"
        f"slurm_job_id={os.getenv('SLURM_JOB_ID', '')}\n"
        f"time={time.time()}\n"
    )

    try:
        # Atomic write: unique temp file in same dir, then rename
        temp_fd, temp_path = tempfile.mkstemp(
            dir=os.path.dirname(marker_path), prefix="spot_signal_tmp_"
        )
        try:
            with os.fdopen(temp_fd, "w") as f:
                f.write(content)
            os.rename(temp_path, marker_path)
        except BaseException:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.debug("Could not create spot signal marker %s: %s", marker_path, e)
        return None

    logger.debug("Created spot signal marker file: %s", marker_path)
    return marker_path


def check_spot_signal(job_store_id: str, consume: bool = True) -> tuple[bool, int | None]:
    """
    Check if the worker for a Toil job received a spot signal.

    Called by the batch system when a job finishes/dies. On success the
    marker is removed by default (consumed) so it cannot leak into a
    later retry attempt of the same job.

    :param job_store_id: The Toil jobStoreID whose worker to check
    :param consume: Remove the marker after reading (default True)
    :return: (received_signal, signal_number) tuple
    """
    marker_path = _marker_path(job_store_id)

    if not os.path.exists(marker_path):
        return False, None

    signal_number: int | None = None
    try:
        with open(marker_path, "r") as f:
            for line in f:
                if line.startswith("signal="):
                    signal_number = int(line.split("=", 1)[1].strip())
                    break
        # Tolerate legacy markers containing a bare signal number
        if signal_number is None:
            with open(marker_path, "r") as f:
                signal_number = int(f.read().strip())
    except Exception as e:
        logger.debug("Could not read spot signal marker %s: %s", marker_path, e)

    if consume:
        try:
            os.remove(marker_path)
        except OSError as e:
            logger.debug("Could not remove spot signal marker %s: %s", marker_path, e)

    if signal_number is None:
        return False, None

    logger.debug(
        "Found spot signal marker for job %s: signal %d", job_store_id, signal_number
    )
    return True, signal_number


def gc_spot_signal_markers(max_age_seconds: float = 7 * 24 * 3600) -> int:
    """
    Delete spot-signal marker files older than ``max_age_seconds``.

    Markers are normally consumed by the batch system or cleared by the next
    attempt of the same job, but a worker SIGKILLed after writing one (or a
    job that is never retried) leaves it behind forever. Called once at batch
    system startup. Returns the number of markers removed.
    """
    marker_dir = _MARKER_DIR or _resolve_marker_dir()
    cutoff = time.time() - max_age_seconds
    removed = 0
    try:
        entries = os.scandir(marker_dir)
    except OSError:
        return 0
    with entries:
        for entry in entries:
            if not entry.name.startswith("spot_signal_"):
                continue
            try:
                if entry.stat().st_mtime < cutoff:
                    os.remove(entry.path)
                    removed += 1
            except OSError:
                continue
    if removed:
        logger.debug("Removed %d stale spot signal markers", removed)
    return removed


def clear_spot_signal(job_store_id: str) -> None:
    """
    Remove any marker file for a Toil job, if present.

    Workers call this at startup so a marker left by a crashed previous
    attempt of the same job cannot be misattributed to the new attempt.
    """
    marker_path = _marker_path(job_store_id)
    try:
        os.remove(marker_path)
        logger.debug("Cleared stale spot signal marker %s", marker_path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.debug("Could not clear spot signal marker %s: %s", marker_path, e)