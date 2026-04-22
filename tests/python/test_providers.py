from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python"

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from voicelayer_orchestrator.providers import reclaim_stale_lock  # noqa: E402


class ReclaimStaleLockTest(unittest.TestCase):
    """Covers :func:`reclaim_stale_lock` which both autostart paths rely on."""

    def setUp(self) -> None:
        super().setUp()
        self.tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="voicelayer-lock-test-"))
        self.lock_path = self.tmp_dir / "endpoint.lock"

    def test_returns_false_when_lock_absent(self) -> None:
        assert not self.lock_path.exists()
        self.assertFalse(reclaim_stale_lock(self.lock_path, "llama-server"))

    def test_leaves_lock_intact_when_owner_still_runs(self) -> None:
        # The current test process is guaranteed to be alive; write its PID
        # and claim its own argv[0] as the expected binary. argv[0] is
        # usually /usr/bin/python or similar; reading /proc/self/cmdline
        # tells us what the file-name comparison should match.
        my_pid = os.getpid()
        cmdline = pathlib.Path(f"/proc/{my_pid}/cmdline").read_bytes()
        argv0 = cmdline.split(b"\x00", 1)[0].decode("utf-8")
        binary_name = pathlib.Path(argv0).name
        self.lock_path.write_text(str(my_pid), encoding="utf-8")

        self.assertFalse(reclaim_stale_lock(self.lock_path, binary_name))
        self.assertTrue(self.lock_path.exists())

    def test_removes_lock_when_owner_pid_is_long_gone(self) -> None:
        # PID 0x7FFFFFFE is almost certainly not a live process on Linux;
        # at minimum /proc/<pid>/cmdline is unreadable, so the helper
        # should treat the lock as stale and delete it.
        self.lock_path.write_text("2147483646", encoding="utf-8")

        self.assertTrue(reclaim_stale_lock(self.lock_path, "llama-server"))
        self.assertFalse(self.lock_path.exists())

    def test_removes_lock_when_pid_runs_different_binary(self) -> None:
        # Pair our own PID with a binary name we're definitely not running.
        my_pid = os.getpid()
        self.lock_path.write_text(str(my_pid), encoding="utf-8")

        self.assertTrue(
            reclaim_stale_lock(self.lock_path, "definitely-not-our-binary-42"),
        )
        self.assertFalse(self.lock_path.exists())

    def test_keeps_lock_when_pid_is_unparseable(self) -> None:
        # Simulates a partial write by the owning process: lock file
        # exists but PID hasn't landed yet. The helper should err on the
        # safe side and leave the lock intact.
        self.lock_path.write_text("", encoding="utf-8")
        self.assertFalse(reclaim_stale_lock(self.lock_path, "llama-server"))
        self.assertTrue(self.lock_path.exists())

        self.lock_path.write_text("not-a-pid", encoding="utf-8")
        self.assertFalse(reclaim_stale_lock(self.lock_path, "llama-server"))
        self.assertTrue(self.lock_path.exists())


if __name__ == "__main__":
    unittest.main()
