"""Provider adapters for the VoiceLayer Python worker."""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import psutil

from voicelayer_orchestrator.config import (
    load_llm_provider_config,
    load_mimo_asr_config,
    load_qwen3_asr_config,
    load_whisper_provider_config,
)


class ProviderInvocationError(RuntimeError):
    """Raised when the configured provider cannot satisfy a request."""


class ProviderUnavailableError(ProviderInvocationError):
    """Raised when no provider is configured for the requested workflow."""


class InvalidProviderParamsError(ProviderInvocationError):
    """Raised when provider routing parameters are malformed (e.g. a
    non-string ``provider_id``)."""


def _is_decorative_annotation(line: str) -> bool:
    """True when a transcript line is a whisper decorative annotation."""

    return (line.startswith("(") and line.endswith(")")) or (
        line.startswith("[") and line.endswith("]")
    )


def collapse_nonspeech_transcript(text: str) -> str:
    """Return an empty string if ``text`` is entirely whisper decorative
    annotations; otherwise return ``text`` unchanged.

    Handles both single-line cases (`(speaks in foreign language)`,
    `(dramatic music)`, `[BLANK_AUDIO]`) and multi-line blocks where
    every non-empty line is itself an annotation. Mixed transcripts —
    any line containing real speech — are preserved verbatim to avoid
    silently dropping substance.
    """

    stripped_lines = [line.strip() for line in text.splitlines()]
    candidate_lines = [line for line in stripped_lines if line]
    if not candidate_lines:
        return ""
    if all(_is_decorative_annotation(line) for line in candidate_lines):
        return ""
    return text


def provider_runtime_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Return the runtime directory used for provider state files."""

    source = environ or os.environ
    base = source.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    runtime_dir = Path(base) / "voicelayer" / "providers"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _read_pid_from_lock(lock_path: Path) -> int | None:
    """Read the PID written by the lock owner.

    Returns ``None`` when the file is missing, unreadable, partially
    written (empty), non-integer, or contains a non-positive value. The
    latter is treated as "unknown" so a malformed file never routes into
    the liveness probe against ``/proc/0`` or negative PIDs.
    """

    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    if not raw:
        return None
    try:
        pid = int(raw)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _pid_runs_binary(pid: int, expected_binary: str) -> bool:
    """Check that ``pid`` still runs ``expected_binary`` (cross-platform).

    Uses psutil's process table (works on Linux and macOS) and compares the
    basename of argv[0] to the basename of ``expected_binary`` so both
    absolute paths and bare names resolve correctly. Returns False on any
    failure (process gone, unreadable cmdline, binary mismatch).
    """

    if pid <= 0:
        return False
    try:
        cmdline = psutil.Process(pid).cmdline()
    except (psutil.Error, OSError):
        return False
    if not cmdline or not cmdline[0]:
        return False
    return Path(cmdline[0]).name == Path(expected_binary).name


def reclaim_stale_lock(lock_path: Path, expected_binary: str) -> bool:
    """Remove ``lock_path`` if its recorded owner is no longer running.

    Returns True when a stale lock was deleted so the caller can retry
    the ``os.open(..., O_EXCL)`` happy path. Returns False when the lock
    owner is still alive (the caller must wait) or when the state is
    ambiguous (partially-written PID, cmdline unreadable) so we err on the
    safe side and leave the lock intact.
    """

    if not lock_path.exists():
        return False
    pid = _read_pid_from_lock(lock_path)
    if pid is None:
        # The owner may have just created the lock and not yet written
        # its PID. Leave the lock alone; the caller will fall through to
        # the shared `wait_for_endpoint` path.
        return False
    if _pid_runs_binary(pid, expected_binary):
        return False
    with contextlib.suppress(FileNotFoundError):
        lock_path.unlink()
    return True


def supported_providers(environ: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Return provider descriptors for the Python worker boundary.

    ``environ`` is the `VOICELAYER_*` override layer; when omitted the
    process environment is read (standalone worker / tests).
    """

    from voicelayer_orchestrator.providers.llm_openai_compatible import (
        configured_llm_descriptor,
    )

    whisper = load_whisper_provider_config(environ)
    mimo = load_mimo_asr_config(environ)
    qwen3 = load_qwen3_asr_config(environ)
    providers: list[dict[str, Any]] = [
        {
            "id": "whisper_cpp",
            "kind": "asr",
            "transport": "whisper_cli" if whisper is not None else "stdio_worker",
            "local": True,
            "default_enabled": True,
            "experimental": False,
            "license": "MIT",
        },
        {
            "id": "voxtral_realtime",
            "kind": "asr",
            "transport": "stdio_worker",
            "local": True,
            "default_enabled": False,
            "experimental": True,
            "license": "Apache-2.0",
        },
        {
            "id": "mimo_v2_5_asr",
            "kind": "asr",
            "transport": "in_process_torch" if mimo is not None else "stdio_worker",
            "local": True,
            "default_enabled": False,
            "experimental": True,
            "license": "MIT",
        },
        {
            "id": "qwen3_asr_1_7b",
            "kind": "asr",
            "transport": "in_process_torch" if qwen3 is not None else "stdio_worker",
            "local": True,
            "default_enabled": False,
            "experimental": True,
            "license": "Apache-2.0",
        },
    ]

    configured = configured_llm_descriptor(load_llm_provider_config(environ))
    if configured is not None:
        providers.append(configured)
    else:
        providers.append(
            {
                "id": "gemma_4_local",
                "kind": "llm",
                "transport": "stdio_worker",
                "local": True,
                "default_enabled": True,
                "experimental": False,
                "license": "Apache-2.0",
            }
        )

    return providers
