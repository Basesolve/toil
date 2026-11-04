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
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inspect the configured batch system and its partition classification."""
import logging
import os

from toil.batchSystems.registry import DEFAULT_BATCH_SYSTEM, get_batch_system
from toil.common import parser_with_common_options
from toil.statsAndLogging import set_logging_from_options

logger = logging.getLogger(__name__)


def _classification_inputs() -> dict[str, str]:
    """Resolve the effective Slurm partition-classification inputs (CLI > env)."""
    try:
        from toil.batchSystems.slurm_mount_recovery import DEFAULT_SPOT_FEATURE
    except ImportError:  # pragma: no cover - slurm always installed with toil
        DEFAULT_SPOT_FEATURE = "spot"

    def pick(*values: str | None, default: str = "(unset)") -> str:
        for value in values:
            if value:
                return str(value)
        return default

    return {
        "spot partitions (globs)": pick(os.getenv("TOIL_SLURM_SPOT_PARTITIONS")),
        "on-demand partitions (globs)": pick(os.getenv("TOIL_SLURM_ONDEMAND_PARTITIONS")),
        "spot node feature": pick(
            os.getenv("TOIL_SLURM_SPOT_FEATURE"), default=DEFAULT_SPOT_FEATURE
        ),
        "legacy name pattern": pick(
            os.getenv("TOIL_SLURM_PARTITION_PREFERED"),
            os.getenv("TOIL_SLURM_PARTITON_PREFERED"),
        ),
        "preference partition type": pick(
            os.getenv("TOIL_SLURM_PREFERENCE_PARTITION_TYPE"), default="spot"
        ),
    }


def check(batch_system_name: str) -> None:
    """Print the batch system kind and, if supported, its partition classification."""
    batch_system = get_batch_system(batch_system_name)
    print(f"Batch system: {batch_system_name} ({batch_system.__name__})")

    assess = getattr(batch_system, "assessBatchResources", None)
    if not callable(assess):
        print("This batch system has no partition classification support.")
        return

    print("\nEffective classification inputs:")
    for key, value in _classification_inputs().items():
        print(f"  {key}: {value}")

    try:
        resources = assess()
    except Exception as e:
        print(f"\nCould not assess batch resources (is {batch_system_name} available here?): {e}")
        return

    print("\nPartitions:")
    if "preference" in getattr(resources, "columns", []):
        for _, row in resources.iterrows():
            print(
                f"  {row['partitions']}: "
                f"preference={'spot' if row['preference'] else 'on_demand'}"
            )
    else:
        for partition in resources["partitions"]:
            print(f"  {partition}: (no classification)")


def main() -> None:
    parser = parser_with_common_options(jobstore_option=False, prog="toil cluster")
    parser.add_argument(
        "--batchSystem",
        dest="batchSystem",
        default=DEFAULT_BATCH_SYSTEM,
        help=f"Batch system to inspect. Default = {DEFAULT_BATCH_SYSTEM}",
    )
    parser.add_argument(
        "subcommand",
        choices=["check"],
        help="check: print the batch system kind and its partition classification.",
    )
    options = parser.parse_args()
    set_logging_from_options(options)

    if options.subcommand == "check":
        check(getattr(options, "batchSystem", None) or DEFAULT_BATCH_SYSTEM)
