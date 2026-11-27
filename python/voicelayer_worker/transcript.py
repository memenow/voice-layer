"""Transcript cleanup helpers shared across ASR providers."""

from __future__ import annotations


def _is_decorative_annotation(line: str) -> bool:
    return (line.startswith("(") and line.endswith(")")) or (
        line.startswith("[") and line.endswith("]")
    )


def collapse_nonspeech_transcript(text: str) -> str:
    """Return an empty string if every non-empty line is a bracketed/parenthetical annotation.

    Mixed transcripts containing at least one real speech line are returned unchanged.
    """

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    if all(_is_decorative_annotation(line) for line in lines):
        return ""
    return text
