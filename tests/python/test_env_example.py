"""Bidirectional coverage guards for ``systemd/voicelayerd.env.example``."""

from __future__ import annotations

import pathlib
import re
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python" / "voicelayer_worker"
RUST_DAEMON_ROOTS: tuple[pathlib.Path, ...] = (
    PROJECT_ROOT / "crates" / "voicelayer-core" / "src",
    PROJECT_ROOT / "crates" / "voicelayerd" / "src",
    PROJECT_ROOT / "crates" / "voicelayer-cli" / "src",
)
ENV_EXAMPLE = PROJECT_ROOT / "systemd" / "voicelayerd.env.example"

CLIENT_ONLY_ENV_VARS: frozenset[str] = frozenset(
    {
        "VOICELAYER_LOG",
        "VOICELAYER_DAEMON_BIN",
    }
)

ENV_VAR_LITERAL_PATTERN = re.compile(r"['\"](VOICELAYER(?:_[A-Z][A-Z0-9]*)+)['\"]")
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


def _collect_literal_references(paths: list[pathlib.Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        names.update(_extract_env_var_literals(path.read_text(encoding="utf-8")))
    return names


class EnvExampleCoverageTest(unittest.TestCase):
    def test_env_example_file_exists(self) -> None:
        self.assertTrue(ENV_EXAMPLE.is_file(), f"missing {ENV_EXAMPLE}")

    def test_every_python_and_rust_daemon_env_var_is_declared_in_env_example(self) -> None:
        env_vars = _extract_env_example_keys(ENV_EXAMPLE.read_text(encoding="utf-8"))
        python_vars = _collect_literal_references(sorted(PYTHON_ROOT.rglob("*.py")))
        rust_vars = _collect_literal_references(
            [p for root in RUST_DAEMON_ROOTS for p in sorted(root.rglob("*.rs"))]
        )
        rust_vars -= CLIENT_ONLY_ENV_VARS

        self.assertEqual(sorted(python_vars - env_vars), [])
        self.assertEqual(sorted(rust_vars - env_vars), [])

        referenced = python_vars | rust_vars | CLIENT_ONLY_ENV_VARS
        self.assertEqual(sorted(env_vars - referenced), [])


class EnvScannerHelpersTest(unittest.TestCase):
    def test_literal_and_assignment_patterns(self) -> None:
        self.assertEqual(
            _extract_env_var_literals('std::env::var("VOICELAYER_FOO") // VOICELAYER_BAR'),
            {"VOICELAYER_FOO"},
        )
        self.assertEqual(_extract_env_var_literals('"VOICELAYER_TRAILING_"'), set())
        self.assertEqual(
            _extract_env_example_keys(
                "VOICELAYER_A=1\n#VOICELAYER_B=2 # inline\n# VOICELAYER_PROSE only\n"
            ),
            {"VOICELAYER_A", "VOICELAYER_B"},
        )
