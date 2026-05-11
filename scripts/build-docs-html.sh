#!/usr/bin/env bash
# Re-render every docs/*.md to a sibling .html using pandoc.
#
# VoiceLayer keeps Markdown as the authoritative source for every page
# under docs/ (the guard tests in `crates/voicelayer-doc-test-utils` +
# `voicelayer-core` / `voicelayerd` / `vl` scan only .md). The HTML
# mirrors are what operators open in a browser. This script is the
# convenience wrapper for regenerating those mirrors after any .md
# change. The CSS, header, and footer partials live under
# `docs/assets/`.
#
# Usage: scripts/build-docs-html.sh
#
# Requirements:
#   pandoc >= 3.0
#   sed (BSD or GNU)
#
# Exits non-zero on the first failure so CI / pre-commit hooks can
# treat a stale mirror as an actionable error.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "build-docs-html: pandoc not found in PATH" >&2
  exit 1
fi

ASSETS_DIR="docs/assets"
HEADER_SUB="${ASSETS_DIR}/header-sub.html"
FOOTER="${ASSETS_DIR}/footer.html"
CSS_REL="../assets/style.css"

if [[ ! -f "$HEADER_SUB" || ! -f "$FOOTER" ]]; then
  echo "build-docs-html: missing partials under $ASSETS_DIR" >&2
  exit 1
fi

render() {
  local src="$1"
  local title="$2"
  local dst="${src%.md}.html"

  pandoc "$src" \
    --standalone \
    --from gfm \
    --to html5 \
    --metadata title="${title} — VoiceLayer" \
    --metadata pagetitle="$title" \
    --css "$CSS_REL" \
    --toc \
    --toc-depth=3 \
    --include-before-body "$HEADER_SUB" \
    --include-after-body "$FOOTER" \
    -o "$dst"
  echo "wrote $dst"
}

render docs/architecture/overview.md                "Architecture Overview"
render docs/architecture/host-injection-strategy.md "Host Injection Strategy"
render docs/architecture/python-worker-protocol.md  "Python Worker Protocol"
render docs/guides/development.md                   "Development Guide"
render docs/guides/desktop.md                       "Desktop Shell Guide"
render docs/guides/local-asr-provider.md            "Local ASR Provider Guide"
render docs/guides/local-llm-provider.md            "Local LLM Provider Guide"
render docs/guides/systemd.md                       "Systemd User Service Guide"

# Rewrite href attributes that still point at the original .md source.
# Inline <code>foo.md</code> mentions (path references in prose) are
# preserved — only href targets are remapped.
find docs -name '*.html' -print0 |
  xargs -0 sed -i -E 's/href="([^"]*)\.md(#[^"]*)?"/href="\1.html\2"/g'

echo "done — docs/index.html stays hand-written; remaining .html files are pandoc mirrors of their .md siblings"
