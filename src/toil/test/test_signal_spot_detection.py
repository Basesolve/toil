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

"""Unit tests for toil.signal_spot_detection."""

import os
import signal
import subprocess
import sys
import tempfile

import pytest

from toil import signal_spot_detection as ssd
from toil.test import ToilTest


class SignalSpotDetectionTest(ToilTest):
    def setUp(self):
        super().setUp()
        self.marker_dir = tempfile.mkdtemp()
        self._old_env = os.environ.get("TOIL_SPOT_SIGNAL_DIR")
        os.environ["TOIL_SPOT_SIGNAL_DIR"] = self.marker_dir
        # Reset module state between tests
        ssd._SPOT_SIGNAL_RECEIVED = None
        ssd._JOB_STORE_ID = "unknown"
        ssd._MARKER_DIR = None

    def tearDown(self):
        if self._old_env is None:
            os.environ.pop("TOIL_SPOT_SIGNAL_DIR", None)
        else:
            os.environ["TOIL_SPOT_SIGNAL_DIR"] = self._old_env
        super().tearDown()

    def test_initialize_uses_env_dir(self):
        signum = ssd.initialize_spot_signal_handling("jobA")
        assert signum == signal.SIGTERM
        assert ssd._MARKER_DIR == self.marker_dir
        assert ssd._JOB_STORE_ID == "jobA"

    def test_configurable_signal(self):
        os.environ["TOIL_SPOT_SIGNAL_SIG"] = "SIGUSR1"
        try:
            assert ssd._get_spot_signal_signum() == signal.SIGUSR1
        finally:
            del os.environ["TOIL_SPOT_SIGNAL_SIG"]

    def test_invalid_signal_falls_back_to_sigterm(self):
        os.environ["TOIL_SPOT_SIGNAL_SIG"] = "NOT_A_SIGNAL"
        try:
            assert ssd._get_spot_signal_signum() == signal.SIGTERM
        finally:
            del os.environ["TOIL_SPOT_SIGNAL_SIG"]

    def test_marker_path_keyed_by_job_store_id(self):
        ssd.initialize_spot_signal_handling("jobB")
        path_a = ssd._marker_path("jobB")
        path_b = ssd._marker_path("jobB")
        path_c = ssd._marker_path("jobC")
        assert path_a == path_b  # deterministic
        assert path_a != path_c  # distinct per job
        assert os.path.dirname(path_a) == self.marker_dir

    def test_report_and_check_roundtrip(self):
        ssd.initialize_spot_signal_handling("jobD")
        ssd._SPOT_SIGNAL_RECEIVED = signal.SIGTERM
        marker = ssd.report_spot_signal(os.getpid())
        assert marker is not None
        assert os.path.exists(marker)
        signaled, signum = ssd.check_spot_signal("jobD")
        assert signaled
        assert signum == signal.SIGTERM
        # Consumed after check
        assert not os.path.exists(marker)

    def test_report_without_signal_returns_none(self):
        ssd.initialize_spot_signal_handling("jobE")
        assert ssd.report_spot_signal(os.getpid()) is None

    def test_check_missing_marker(self):
        ssd.initialize_spot_signal_handling("jobF")
        assert ssd.check_spot_signal("jobF") == (False, None)

    def test_check_no_consume(self):
        ssd.initialize_spot_signal_handling("jobG")
        ssd._SPOT_SIGNAL_RECEIVED = signal.SIGTERM
        ssd.report_spot_signal(os.getpid())
        signaled, _ = ssd.check_spot_signal("jobG", consume=False)
        assert signaled
        # Still there
        signaled2, _ = ssd.check_spot_signal("jobG")
        assert signaled2

    def test_legacy_bare_number_marker(self):
        ssd.initialize_spot_signal_handling("jobH")
        marker = ssd._marker_path("jobH")
        with open(marker, "w") as f:
            f.write("15")
        signaled, signum = ssd.check_spot_signal("jobH")
        assert signaled
        assert signum == 15

    def test_clear_stale_marker(self):
        ssd.initialize_spot_signal_handling("jobI")
        marker = ssd._marker_path("jobI")
        with open(marker, "w") as f:
            f.write("signal=15\n")
        ssd.clear_spot_signal("jobI")
        assert not os.path.exists(marker)
        # Clearing nonexistent marker is a no-op
        ssd.clear_spot_signal("jobI")

    def test_initialize_clears_stale_marker(self):
        # Simulate a previous attempt that crashed after writing a marker
        ssd.initialize_spot_signal_handling("jobJ")
        marker = ssd._marker_path("jobJ")
        with open(marker, "w") as f:
            f.write("signal=15\n")
        # New attempt of the same job must not see the stale marker
        ssd.initialize_spot_signal_handling("jobJ")
        assert ssd.check_spot_signal("jobJ") == (False, None)

    def test_corrupt_marker_tolerated(self):
        ssd.initialize_spot_signal_handling("jobK")
        marker = ssd._marker_path("jobK")
        with open(marker, "w") as f:
            f.write("garbage-no-signal-here")
        assert ssd.check_spot_signal("jobK") == (False, None)
        # Corrupt marker is still consumed
        assert not os.path.exists(marker)

    def test_sigterm_handler_writes_marker_and_rekills(self):
        """End-to-end: a child that installs the handler must die by SIGTERM
        and leave a consumable marker keyed by jobStoreID."""
        child_code = (
            "import signal, time, sys\n"
            "from toil import signal_spot_detection as s\n"
            "s.initialize_spot_signal_handling('jobE2E')\n"
            "signal.signal(s._get_spot_signal_signum(), s.handle_spot_signal)\n"
            "sys.stderr.write('READY\\n'); sys.stderr.flush()\n"
            "time.sleep(30)\n"
        )
        env = dict(os.environ, TOIL_SPOT_SIGNAL_DIR=self.marker_dir)
        child = subprocess.Popen(
            [sys.executable, "-c", child_code],
            env=env,
            stderr=subprocess.PIPE,
        )
        try:
            assert child.stderr.readline().decode().strip() == "READY"
            child.send_signal(signal.SIGTERM)
            rc = child.wait()
            # Killed by SIGTERM (not clean exit, not SIGKILL)
            assert rc == -signal.SIGTERM
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
        # Batch system side: find marker without knowing the child PID
        ssd.initialize_spot_signal_handling("other-job")
        signaled, signum = ssd.check_spot_signal("jobE2E")
        assert signaled
        assert signum == signal.SIGTERM


if __name__ == "__main__":
    pytest.main([__file__])
