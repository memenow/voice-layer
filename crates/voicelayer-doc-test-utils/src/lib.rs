//! Shared test helpers for VoiceLayer's repository-wide markdown
//! guard tests.
//!
//! Several `#[cfg(test)]` modules across the workspace need to walk
//! every `.md` file under `docs/` (and a couple of repo-root markdown
//! files such as `README.md`) so they can scan operator-facing prose
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

/// Recursively collect every `.md` file under `start` into `out`.
///
/// Used by guard tests across the workspace to enumerate all
/// operator-facing markdown without each test re-implementing the
/// same recursive walker.
///
/// Files are appended to `out` in the order `read_dir` yields them
/// (filesystem-defined). Callers that need a deterministic order
/// should sort the result themselves; the existing guard tests do
/// not rely on order because they aggregate violations into a
/// `Vec<String>` that they sort or `BTreeSet` afterwards.
///
/// Symlink loops are not detected — the workspace's `docs/` tree
/// does not contain any, and adding cycle detection here would be
/// dead weight.
pub fn collect_markdown_files(start: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(start)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_markdown_files(&path, out)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("md") {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::collect_markdown_files;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn collect_markdown_files_returns_only_md_extensions_across_nested_directories() {
        let tmp = TempDir::new().expect("create tempdir");
        let root = tmp.path();

        // Layout:
        //   root/
        //     top.md
        //     ignore.txt
        //     nested/
        //       inner.md
        //       README           (no extension — must be skipped)
        //       deeper/
        //         leaf.md
        //         leaf.markdown  (NOT `.md` — must be skipped)
        let nested = root.join("nested");
        let deeper = nested.join("deeper");
        fs::create_dir_all(&deeper).expect("create nested dirs");

        for (rel, body) in [
            ("top.md", "# top"),
            ("ignore.txt", "not markdown"),
            ("nested/inner.md", "# inner"),
            ("nested/README", "extensionless"),
            ("nested/deeper/leaf.md", "# leaf"),
            ("nested/deeper/leaf.markdown", "wrong extension"),
        ] {
            fs::write(root.join(rel), body).unwrap_or_else(|err| {
                panic!("write {rel}: {err}");
            });
        }

        let mut collected: Vec<PathBuf> = Vec::new();
        collect_markdown_files(root, &mut collected).expect("walk tempdir");
        collected.sort();

        let expected: Vec<PathBuf> = ["nested/deeper/leaf.md", "nested/inner.md", "top.md"]
            .iter()
            .map(|rel| root.join(rel))
            .collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn collect_markdown_files_appends_to_existing_vec_without_clearing_it() {
        // The guard tests prime `out` with `vec![repo_root.join("README.md")]`
        // before calling the walker; pin that the walker preserves
        // the seed entries instead of overwriting them.
        let tmp = TempDir::new().expect("create tempdir");
        let root = tmp.path();
        fs::write(root.join("only.md"), "# only").expect("write only.md");

        let seed = PathBuf::from("/tmp/sentinel-readme.md");
        let mut collected: Vec<PathBuf> = vec![seed.clone()];
        collect_markdown_files(root, &mut collected).expect("walk tempdir");

        assert!(
            collected.contains(&seed),
            "seed entry was dropped: {collected:?}",
        );
        assert!(
            collected.contains(&root.join("only.md")),
            "discovered .md missing: {collected:?}",
        );
    }
}
