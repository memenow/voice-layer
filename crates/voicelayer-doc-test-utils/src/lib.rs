//! Shared test helpers for VoiceLayer's repository-wide documentation
//! guard tests.
//!
//! Several `#[cfg(test)]` modules across the workspace need to walk
//! every `.html` file under `docs/` (plus a couple of repo-root files
//! such as `README.md`) so they can scan operator-facing prose
//! for drift against the code: route paths, config keys, env vars,
//! cross-reference targets, and so on. Rather than each guard test
//! re-implementing the same recursive walker against `std::fs`, the
//! workers all call into this crate.
//!
//! The crate is `publish = false` and intentionally has zero runtime
//! dependencies — the only thing it pulls in is `tempfile` as a
//! dev-dependency so its own unit test can exercise the walker
//! against a synthetic tree.
//!
//! See `crates/voicelayer-core/src/domain.rs`,
//! `crates/voicelayerd/src/lib.rs`, and `crates/vl/src/config.rs` for
//! the call sites.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Recursively collect every documentation `.html` file under `start`
/// into `out`.
///
/// Used by guard tests across the workspace to enumerate all
/// operator-facing HTML documentation without each test
/// re-implementing the same recursive walker. The `docs/assets/`
/// partials and stylesheet are skipped because they are shared
/// chrome, not standalone documentation pages.
///
/// Files are appended to `out` in the order `read_dir` yields them
/// (filesystem-defined). Callers that need a deterministic order
/// should sort the result themselves; the existing guard tests do
/// not rely on order because they aggregate violations into a
/// `Vec<String>` that they sort or `BTreeSet` afterwards.
///
/// Symlink loops are not detected; the workspace's `docs/` tree
/// does not contain any, and adding cycle detection here would be
/// dead weight.
pub fn collect_html_doc_files(start: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(start)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            if entry.file_name() == "assets" {
                continue;
            }
            collect_html_doc_files(&path, out)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("html") {
            out.push(path);
        }
    }
    Ok(())
}

/// Extract code literals from Markdown inline-code spans and HTML
/// `<code>...</code>` elements.
///
/// The guard tests still scan the repo-root `README.md` as Markdown,
/// while `docs/` pages are now maintained directly as HTML. This
/// helper gives tests that care about code-styled prose one common
/// token stream without pulling in an HTML parser.
pub fn extract_doc_code_literals(contents: &str) -> Vec<String> {
    let mut literals = Vec::new();

    let mut markdown = contents;
    while let Some(idx) = markdown.find('`') {
        let after = &markdown[idx + 1..];
        let Some(close) = after.find('`') else {
            break;
        };
        let inner = &after[..close];
        markdown = &after[close + 1..];
        if !inner.is_empty() {
            literals.push(inner.to_owned());
        }
    }

    let mut html = contents;
    while let Some(idx) = html.find("<code") {
        let after_tag_start = &html[idx + "<code".len()..];
        let Some(tag_end) = after_tag_start.find('>') else {
            break;
        };
        let after_open = &after_tag_start[tag_end + 1..];
        let Some(close) = after_open.find("</code>") else {
            break;
        };
        let inner = &after_open[..close];
        html = &after_open[close + "</code>".len()..];
        if !inner.is_empty() {
            literals.push(decode_minimal_html_entities(inner));
        }
    }

    literals
}

fn decode_minimal_html_entities(input: &str) -> String {
    input
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&#x27;", "'")
        .replace("&amp;", "&")
}

#[cfg(test)]
mod tests {
    use super::{collect_html_doc_files, extract_doc_code_literals};
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn collect_html_doc_files_returns_only_html_pages_across_nested_directories() {
        let tmp = TempDir::new().expect("create tempdir");
        let root = tmp.path();

        // Layout:
        //   root/
        //     index.html
        //     ignore.txt
        //     assets/
        //       footer.html    (shared partial, skipped)
        //     nested/
        //       inner.html
        //       README           (no extension — must be skipped)
        //       deeper/
        //         leaf.html
        //         leaf.xhtml  (NOT `.html` — must be skipped)
        let nested = root.join("nested");
        let deeper = nested.join("deeper");
        fs::create_dir_all(&deeper).expect("create nested dirs");
        fs::create_dir_all(root.join("assets")).expect("create assets dir");

        for (rel, body) in [
            ("index.html", "<h1>top</h1>"),
            ("assets/footer.html", "<footer>shared chrome</footer>"),
            ("ignore.txt", "not markdown"),
            ("nested/inner.html", "<h1>inner</h1>"),
            ("nested/README", "extensionless"),
            ("nested/deeper/leaf.html", "<h1>leaf</h1>"),
            ("nested/deeper/leaf.xhtml", "wrong extension"),
        ] {
            fs::write(root.join(rel), body).unwrap_or_else(|err| {
                panic!("write {rel}: {err}");
            });
        }

        let mut collected: Vec<PathBuf> = Vec::new();
        collect_html_doc_files(root, &mut collected).expect("walk tempdir");
        collected.sort();

        let expected: Vec<PathBuf> = ["index.html", "nested/deeper/leaf.html", "nested/inner.html"]
            .iter()
            .map(|rel| root.join(rel))
            .collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn collect_html_doc_files_appends_to_existing_vec_without_clearing_it() {
        // The guard tests prime `out` with `vec![repo_root.join("README.md")]`
        // before calling the walker; pin that the walker preserves
        // the seed entries instead of overwriting them.
        let tmp = TempDir::new().expect("create tempdir");
        let root = tmp.path();
        fs::write(root.join("only.html"), "<h1>only</h1>").expect("write only.html");

        let seed = PathBuf::from("/tmp/sentinel-readme.md");
        let mut collected: Vec<PathBuf> = vec![seed.clone()];
        collect_html_doc_files(root, &mut collected).expect("walk tempdir");

        assert!(
            collected.contains(&seed),
            "seed entry was dropped: {collected:?}",
        );
        assert!(
            collected.contains(&root.join("only.html")),
            "discovered .html missing: {collected:?}",
        );
    }

    #[test]
    fn extract_doc_code_literals_handles_markdown_and_html_code_spans() {
        let contents = concat!(
            "See `README.md`.\n",
            "<p>Use <code>docs/guides/systemd.html</code>.</p>\n",
            "<pre><code>VOICELAYER_SOCKET_PATH=&quot;/tmp/socket&quot;</code></pre>\n",
        );

        let literals = extract_doc_code_literals(contents);
        assert!(literals.contains(&"README.md".to_owned()));
        assert!(literals.contains(&"docs/guides/systemd.html".to_owned()));
        assert!(literals.contains(&"VOICELAYER_SOCKET_PATH=\"/tmp/socket\"".to_owned()));
    }
}
