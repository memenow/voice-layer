<!-- Keep the summary concise; put deep rationale in commits or docs. -->

## Summary

<!-- One or two sentences describing the user-visible change. -->

## Changes

-

## Verification

Run the verification chain locally with Rust 1.100 and Python 3.12 via `uv`. Check each item or explain any skip.

- [ ] `cargo fmt --all`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo test --all`
- [ ] `uv run ruff check python tests/python`
- [ ] `uv run ruff format --check python tests/python`
- [ ] `uv run pytest -q tests/python`
- [ ] Manual check of the affected runtime path

## Linked issue

<!-- e.g. Closes #123 -->

## Breaking changes

<!-- Public API, CLI flags, config schema, or OpenAPI changes and migration steps. -->

## Risks and follow-ups

<!-- Platform-specific behavior, known gaps, or follow-up work. -->
