"""Regression tests for ``systemd/voicelayerd.env.example``.

The env file and the Python worker drift apart easily: add an env var to
``config.py`` and the template silently falls out of sync, so operators
seeding their systemd unit from the example never know they are missing
a knob. This test locks the invariant: every ``VOICELAYER_*`` env var
referenced by the Python worker must appear as a key in the example
file, either live or commented out.

Rust-side knobs are intentionally *not* scanned — this asserts a
one-directional subset relationship (python ⊆ example) so the example
can legitimately include daemon/CLI/install-script-only vars like
``VOICELAYER_PROJECT_ROOT`` and ``VOICELAYER_WORKER_TIMEOUT_SECONDS``
that no Python module touches.
"""

from __future__ import annotations

import pathlib
import re
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python" / "voicelayer_orchestrator"
ENV_EXAMPLE = PROJECT_ROOT / "systemd" / "voicelayerd.env.example"

# The regex matches only static string literals. Dynamic lookups like
# `source.get(f"VOICELAYER_{suffix}")` would escape this scan; the
# codebase has no such lookups today and adding one should be rare
# enough that hand-maintenance is fine.
ENV_VAR_PATTERN = re.compile(r"VOICELAYER_[A-Z][A-Z0-9_]*")


def _extract_env_var_names(text: str) -> set[str]:
    return set(ENV_VAR_PATTERN.findall(text))


def _list_python_sources() -> list[pathlib.Path]:
    return sorted(PYTHON_ROOT.rglob("*.py"))


class EnvExampleCoverageTest(unittest.TestCase):
    def test_env_example_file_exists(self) -> None:
        self.assertTrue(
            ENV_EXAMPLE.is_file(),
            f"env example template missing at {ENV_EXAMPLE}",
        )

    def test_every_python_env_var_is_declared_in_env_example(self) -> None:
        python_vars: set[str] = set()
        for source in _list_python_sources():
            python_vars.update(_extract_env_var_names(source.read_text(encoding="utf-8")))

        env_vars = _extract_env_var_names(ENV_EXAMPLE.read_text(encoding="utf-8"))
        missing = sorted(python_vars - env_vars)
        self.assertFalse(
            missing,
            (
                "env example is missing vars referenced by the Python worker: "
                f"{missing}. Add them to systemd/voicelayerd.env.example "
                "(commented out if optional) so operators seeding their unit "
                "see every knob."
            ),
        )


if __name__ == "__main__":
    unittest.main()
