<!-- Keep the summary terse; details belong in commits or linked design notes. -->

## Summary

<!-- One or two sentences describing the user-visible change. -->

## Updates

<!-- Bullet list of notable code, doc, or config changes. -->

-

## Verification

Confirm the authoritative verification chain ran locally on Ubuntu with Rust 1.88
and Python 3.12 via uv. Check each box that applied or strike through with an
explanation when intentionally skipped.

- [ ] `cargo fmt --all`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo test --all`
- [ ] `uv run ruff check python tests/python`
- [ ] `uv run ruff format --check python tests/python`
- [ ] `uv run pytest -q tests/python`
- [ ] Manual exercise of the affected path (daemon, CLI, desktop, worker, or docs)

## Linked issue

<!-- e.g. Closes #123, Refs #456. Leave blank if none. -->

## Breaking changes

<!-- Public API, CLI flags, configuration schema, OpenAPI surface, on-disk
     formats. Note migration steps if applicable. -->

## Risks and follow-ups

<!-- Performance, privacy, host integration risks, known gaps, deferred work. -->
