"""Regression tests for ``systemd/voicelayerd.env.example``.

The env file, the Python worker, and the Rust daemon drift apart
easily: add an env var on any of those surfaces and the template
silently falls out of sync, so operators seeding their systemd unit
from the example never know they are missing a knob. This module
locks the invariant from both directions:

- Every ``VOICELAYER_*`` env var looked up from ``python/`` or from
  the Rust daemon crates (``crates/voicelayerd``, ``crates/vl``) must
  appear as a ``KEY=`` or ``#KEY=`` line in the example.
- vl-desktop and install-script-only knobs are intentionally excluded
  via an explicit allow-list so the daemon env file stays scoped to
  the daemon process.
"""

from __future__ import annotations

import pathlib
import re
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python" / "voicelayer_orchestrator"
RUST_DAEMON_ROOTS: tuple[pathlib.Path, ...] = (
    PROJECT_ROOT / "crates" / "voicelayerd" / "src",
    PROJECT_ROOT / "crates" / "vl" / "src",
)
ENV_EXAMPLE = PROJECT_ROOT / "systemd" / "voicelayerd.env.example"

# Client-side env vars that legitimately do not belong in the daemon
# env example. The keys here are read by vl-desktop (an interactive
# GUI process, not a systemd user service) and documented in
# docs/guides/desktop.md. Adding a new client-only knob requires
# updating both this set and the desktop guide in the same PR.
CLIENT_ONLY_ENV_VARS: frozenset[str] = frozenset(
    {
        "VOICELAYER_LOG",
        "VOICELAYER_VL_BIN",
    }
)

# Matches a quoted ``"VOICELAYER_FOO_BAR"`` style literal — the shape
# both ``std::env::var("...")`` in Rust and ``source.get("...")`` in
# Python use to look up env vars. Scanning literals (rather than any
# bare-token mention) keeps prose comments like
# ``// export VOICELAYER_WHISPER_* itself`` from being mistaken for
# real references. Variable names use underscore-separated uppercase
# segments, which matches every real knob and rejects the
# trailing-underscore false positive.
#
# Dynamic lookups (``concat!``, f-strings) would escape this scan;
# the codebase has no such lookups today and adding one should be
# rare enough that hand-maintenance is fine.
ENV_VAR_LITERAL_PATTERN = re.compile(
    r"""
    ['"]                                # opening quote (single or double)
    (VOICELAYER(?:_[A-Z][A-Z0-9]*)+)    # the env var name
    ['"]                                # closing quote
    """,
    re.VERBOSE,
)

# Matches a single assignment-shaped line in the env example: optional
# leading `#` (commented-out override), optional whitespace, the key,
# then `=`. Scanning only these lines is what makes this test actually
# enforce "operators can set this knob" rather than "the name appears
# somewhere in the file" — a name that shows up only in prose would
# otherwise satisfy the check without any `KEY=` entry to uncomment.
ENV_EXAMPLE_ASSIGNMENT_PATTERN = re.compile(r"^\s*#?\s*(VOICELAYER(?:_[A-Z][A-Z0-9]*)+)\s*=")


def _extract_env_var_literals(text: str) -> set[str]:
    return set(ENV_VAR_LITERAL_PATTERN.findall(text))


def _extract_env_example_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for line in text.splitlines():
        match = ENV_EXAMPLE_ASSIGNMENT_PATTERN.match(line)
        if match is not None:
            keys.add(match.group(1))
    return keys


def _list_python_sources() -> list[pathlib.Path]:
    return sorted(PYTHON_ROOT.rglob("*.py"))


def _list_rust_daemon_sources() -> list[pathlib.Path]:
    sources: list[pathlib.Path] = []
    for root in RUST_DAEMON_ROOTS:
        sources.extend(sorted(root.rglob("*.rs")))
    return sources


def _collect_literal_references(paths: list[pathlib.Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        names.update(_extract_env_var_literals(path.read_text(encoding="utf-8")))
    return names


class EnvExampleCoverageTest(unittest.TestCase):
    def test_env_example_file_exists(self) -> None:
        self.assertTrue(
            ENV_EXAMPLE.is_file(),
            f"env example template missing at {ENV_EXAMPLE}",
        )

    def test_every_python_env_var_is_declared_in_env_example(self) -> None:
        python_vars = _collect_literal_references(_list_python_sources())
        env_vars = _extract_env_example_keys(ENV_EXAMPLE.read_text(encoding="utf-8"))
        missing = sorted(python_vars - env_vars)
        self.assertFalse(
            missing,
            (
                "env example is missing vars referenced by the Python worker: "
                f"{missing}. Add them to systemd/voicelayerd.env.example as a "
                "`KEY=` or `#KEY=` line (prose mentions do not count) so "
                "operators seeding their unit see every knob."
            ),
        )

    def test_every_rust_daemon_env_var_is_declared_in_env_example(self) -> None:
        rust_vars = _collect_literal_references(_list_rust_daemon_sources())
        rust_vars -= CLIENT_ONLY_ENV_VARS

        env_vars = _extract_env_example_keys(ENV_EXAMPLE.read_text(encoding="utf-8"))
        missing = sorted(rust_vars - env_vars)
        self.assertFalse(
            missing,
            (
                "env example is missing vars referenced by the Rust daemon: "
                f"{missing}. Add them to systemd/voicelayerd.env.example as a "
                "`KEY=` or `#KEY=` line — or, if the knob is legitimately "
                "client-side-only (read by vl-desktop, not the daemon), add "
                "it to CLIENT_ONLY_ENV_VARS in this test and document it in "
                "docs/guides/desktop.md."
            ),
        )


class EnvVarLiteralPatternTest(unittest.TestCase):
    """Proves the literal scanner only counts quoted string literals."""

    def test_double_quoted_literal_is_extracted(self) -> None:
        self.assertEqual(
            _extract_env_var_literals('std::env::var("VOICELAYER_FOO_BAR")'),
            {"VOICELAYER_FOO_BAR"},
        )

    def test_single_quoted_literal_is_extracted(self) -> None:
        self.assertEqual(
            _extract_env_var_literals("source.get('VOICELAYER_FOO_BAR')"),
            {"VOICELAYER_FOO_BAR"},
        )

    def test_prose_mention_is_ignored(self) -> None:
        # Matches the real-world false positive that motivated the
        # literal-scoped scan: `// export VOICELAYER_WHISPER_* itself.`
        text = "// Operators should export VOICELAYER_WHISPER_* themselves."
        self.assertEqual(_extract_env_var_literals(text), set())

    def test_trailing_underscore_is_not_captured(self) -> None:
        self.assertEqual(
            _extract_env_var_literals('"VOICELAYER_WHISPER_"'),
            set(),
            "trailing underscore is not a valid env var name shape",
        )


class EnvExampleParserTest(unittest.TestCase):
    """Proves the env example parser only counts assignment-shaped lines."""

    def test_live_assignment_is_extracted(self) -> None:
        self.assertEqual(
            _extract_env_example_keys("VOICELAYER_FOO=bar"),
            {"VOICELAYER_FOO"},
        )

    def test_commented_out_assignment_is_extracted(self) -> None:
        self.assertEqual(
            _extract_env_example_keys("#VOICELAYER_FOO=bar"),
            {"VOICELAYER_FOO"},
        )
        self.assertEqual(
            _extract_env_example_keys("#  VOICELAYER_FOO=bar"),
            {"VOICELAYER_FOO"},
        )

    def test_prose_mention_is_ignored(self) -> None:
        text = (
            "# VOICELAYER_FOO is documented here but never written as an\n"
            "# assignment — operators have no `KEY=` line to uncomment.\n"
        )
        self.assertEqual(_extract_env_example_keys(text), set())

    def test_assignment_with_inline_comment_is_extracted(self) -> None:
        self.assertEqual(
            _extract_env_example_keys("VOICELAYER_FOO=bar # inline"),
            {"VOICELAYER_FOO"},
        )


if __name__ == "__main__":
    unittest.main()
