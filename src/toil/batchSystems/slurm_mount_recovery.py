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
Helpers for Slurm mount / storage I/O failure detection and recovery.
"""
from __future__ import annotations

import errno
import logging
import os
import re
from typing import TYPE_CHECKING

from toil.lib.misc import CalledProcessErrorStderr, call_command

if TYPE_CHECKING:
    from toil.batchSystems.slurm import SlurmBatchSystem

logger = logging.getLogger(__name__)

# Worker exit code reserved for node-local storage I/O failures (not 137 OOM / 143 SIGTERM).
STORAGE_FAILURE_EXIT_CODE = 136

STORAGE_IO_ERRNOS = frozenset({errno.EIO, 5})

STORAGE_FAILURE_LOG_MARKERS = (
    "Input/output error",
    "OSError: [Errno 5]",
    "[Errno 5] Input/output error",
)

# Slurm job states where in-place partition update is still meaningful.
PARTITION_SWITCH_STATES = frozenset(
    {
        "PENDING",
        "REQUEUE_HOLD",
        "REQUEUED",
        "SUSPENDED",
        "RUNNING",
    }
)

DEFAULT_PARTITION_SWITCH_COOLDOWN = 300
DEFAULT_PARTITION_SWITCH_POLL_INTERVAL = 0.25
DEFAULT_MAX_EXCLUDED_NODES = 64
DEFAULT_LOST_JOB_TIMEOUT: float | None = None


def env_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def env_csv(name: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def is_fatal_storage_oserror(
    exc: BaseException, protected_paths: list[str | None]
) -> bool:
    """
    Return True if exc is an OSError indicating I/O failure on a protected path.
    """
    if not isinstance(exc, OSError):
        return False
    if exc.errno not in STORAGE_IO_ERRNOS:
        return False
    normalized = [
        os.path.abspath(p)
        for p in protected_paths
        if p is not None and p != ""
    ]
    if not normalized:
        return True
    for path in normalized:
        filename = getattr(exc, "filename", None)
        if filename is not None:
            try:
                if os.path.abspath(filename).startswith(path + os.sep) or os.path.abspath(
                    filename
                ) == path:
                    return True
            except OSError:
                continue
    return False


def storage_failure_in_text(text: str) -> bool:
    return any(marker in text for marker in STORAGE_FAILURE_LOG_MARKERS)


def _split_slurm_list_at_depth_zero(value: str) -> list[str]:
    """Split a Slurm list on commas that appear outside of ``[...]`` bracket groups."""
    segments: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(value):
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            segments.append(value[start:index])
            start = index + 1
    segments.append(value[start:])
    return segments


def _expand_slurm_bracket_token(prefix: str, token: str, nodes: set[str]) -> None:
    """Expand one inner bracket token (e.g. ``001-003`` or ``005``) into hostnames."""
    token = token.strip()
    if not token:
        return
    range_match = re.match(r"^(\d+)(?:-(\d+))?$", token)
    if not range_match:
        nodes.add(f"{prefix}{token}")
        return
    start, end = range_match.group(1), range_match.group(2)
    if end is None:
        nodes.add(f"{prefix}{start}")
        return
    width = len(start)
    for i in range(int(start), int(end) + 1):
        nodes.add(f"{prefix}{str(i).zfill(width)}")


def _parse_slurm_nodelist_local(nodelist: str) -> set[str]:
    """Parse a NodeList without calling Slurm (comma-safe inside brackets)."""
    nodes: set[str] = set()
    for segment in _split_slurm_list_at_depth_zero(nodelist):
        segment = segment.strip()
        if not segment:
            continue
        bracket_match = re.match(r"^([^\[]+)\[(.+)\]$", segment)
        if bracket_match:
            prefix, inner = bracket_match.group(1), bracket_match.group(2)
            for token in _split_slurm_list_at_depth_zero(inner):
                _expand_slurm_bracket_token(prefix, token, nodes)
        else:
            nodes.add(segment)
    return nodes


def parse_slurm_nodelist(nodelist: str | None) -> set[str]:
    """
    Parse Slurm NodeList / BatchHost values into individual node hostnames.
    """
    if not nodelist:
        return set()
    nodelist = nodelist.strip()
    try:
        stdout = run_scontrol("show", "hostnames", nodelist, quiet=True)
        nodes = {line.strip() for line in stdout.splitlines() if line.strip()}
        if nodes:
            return nodes
    except (CalledProcessErrorStderr, OSError):
        pass
    return _parse_slurm_nodelist_local(nodelist)


def build_scontrol_argv(*args: str) -> list[str]:
    prefix = os.getenv("TOIL_SLURM_SCONTROL_PREFIX", "").strip()
    scontrol = os.getenv("TOIL_SLURM_SCONTROL", "scontrol")
    cmd: list[str] = []
    if prefix:
        cmd.extend(prefix.split())
    cmd.append(scontrol)
    cmd.extend(args)
    return cmd


def run_scontrol(*args: str, quiet: bool = False) -> str:
    return call_command(build_scontrol_argv(*args), quiet=quiet)


def parse_scontrol_job_lines(lines: list[str]) -> dict[str, str]:
    """Parse ``scontrol -o show job`` line-oriented key=value output."""
    job: dict[str, str] = {}
    key = ""
    for item in lines:
        bits = item.split("=", 1)
        if len(bits) == 1:
            if key:
                job[key] += " " + bits[0]
        else:
            key = bits[0]
            job[key] = bits[1]
    return job


def partition_switch_reason_matches(reason: str | None) -> bool:
    if not reason:
        return True
    patterns = env_csv("TOIL_SLURM_PARTITION_SWITCH_REASONS")
    if not patterns:
        return True
    reason_lower = reason.lower()
    return any(p.lower() in reason_lower for p in patterns)


def batch_logs_indicate_storage_failure(
    boss: SlurmBatchSystem, toil_job_id: int, slurm_job_id: int
) -> bool:
    """Scan this Slurm attempt's stdout/stderr logs for I/O error markers."""
    for std in ("out", "err"):
        try:
            path = boss.format_std_out_err_path(
                toil_job_id, str(slurm_job_id), std
            )
        except Exception:
            continue
        try:
            with open(path, "rb") as handle:
                chunk = handle.read(65536)
            if storage_failure_in_text(chunk.decode("utf-8", errors="replace")):
                return True
        except OSError:
            continue
    return False
