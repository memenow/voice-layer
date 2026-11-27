//! Shared test utilities for VoiceLayer workspace tests.
//!
//! Provides documentation file discovery, inline code-literal extraction, and
//! a mutex-serialized environment variable guard for tests that mutate process
//! environment state.

use std::ffi::{OsStr, OsString};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Recursively collect `.html` files under `start` into `out`, skipping `assets/`.
///
/// Entries are appended in filesystem order without clearing existing items in `out`.
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

/// Extract code literals from Markdown inline code spans and HTML `<code>...</code>` tags.
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

/// Process-wide lock that serializes tests mutating environment variables.
pub fn env_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Scoped environment variable guard that restores prior values on drop.
///
/// Holds the process-wide [`env_lock`] for its lifetime so concurrent tests do
/// not race on `std::env`.
pub struct EnvGuard {
    _lock: MutexGuard<'static, ()>,
    saved: Vec<(OsString, Option<OsString>)>,
}

impl EnvGuard {
    /// Acquire the process-wide environment lock and return an empty guard.
    pub fn lock() -> Self {
        Self {
            _lock: env_lock(),
            saved: Vec::new(),
        }
    }

    /// Set `key` to `value` for the lifetime of this guard.
    pub fn set(&mut self, key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) {
        let key_os = key.as_ref().to_os_string();
        self.record_original(&key_os);
        // SAFETY: Serialized by the process-wide mutex held by `self._lock`.
        unsafe { std::env::set_var(&key_os, value) };
    }

    /// Remove `key` for the lifetime of this guard.
    pub fn remove(&mut self, key: impl AsRef<OsStr>) {
        let key_os = key.as_ref().to_os_string();
        self.record_original(&key_os);
        // SAFETY: Serialized by the process-wide mutex held by `self._lock`.
        unsafe { std::env::remove_var(&key_os) };
    }

    fn record_original(&mut self, key: &OsStr) {
        if !self.saved.iter().any(|(existing, _)| existing == key) {
            self.saved.push((key.to_os_string(), std::env::var_os(key)));
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, prior) in self.saved.drain(..).rev() {
            match prior {
                // SAFETY: Still holding `self._lock` until `Drop` finishes.
                Some(value) => unsafe { std::env::set_var(&key, value) },
                None => unsafe { std::env::remove_var(&key) },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EnvGuard, collect_html_doc_files, extract_doc_code_literals};
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn collect_html_doc_files_returns_only_html_pages_across_nested_directories() {
        let tmp = TempDir::new().expect("create tempdir");
        let root = tmp.path();

        let nested = root.join("nested");
        let deeper = nested.join("deeper");
        fs::create_dir_all(&deeper).expect("create nested dirs");
        fs::create_dir_all(root.join("assets")).expect("create assets dir");

        for (rel, body) in [
            ("index.html", "<h1>top</h1>"),
            ("assets/partial.html", "<footer>shared chrome</footer>"),
            ("ignore.txt", "not html"),
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
        let tmp = TempDir::new().expect("create tempdir");
        let root = tmp.path();
        fs::write(root.join("only.html"), "<h1>only</h1>").expect("write only.html");

        let seed = PathBuf::from("/tmp/sentinel-readme.md");
        let mut collected: Vec<PathBuf> = vec![seed.clone()];
        collect_html_doc_files(root, &mut collected).expect("walk tempdir");

        assert!(collected.contains(&seed));
        assert!(collected.contains(&root.join("only.html")));
    }

    #[test]
    fn extract_doc_code_literals_handles_markdown_and_html_code_spans() {
        let contents = concat!(
            "See `README.md`.\n",
            "<p>Use <code>docs/guides/services.html</code>.</p>\n",
            "<pre><code>VOICELAYER_SOCKET_PATH=&quot;/tmp/socket&quot;</code></pre>\n",
        );

        let literals = extract_doc_code_literals(contents);
        assert!(literals.contains(&"README.md".to_owned()));
        assert!(literals.contains(&"docs/guides/services.html".to_owned()));
        assert!(literals.contains(&"VOICELAYER_SOCKET_PATH=\"/tmp/socket\"".to_owned()));
    }

    #[test]
    fn env_guard_restores_prior_values_on_drop() {
        const KEY: &str = "VOICELAYER_TESTKIT_ENV_GUARD_PROBE";
        {
            let mut env = EnvGuard::lock();
            env.set(KEY, "first");
            assert_eq!(std::env::var(KEY).as_deref(), Ok("first"));
            env.set(KEY, "second");
            assert_eq!(std::env::var(KEY).as_deref(), Ok("second"));
            env.remove(KEY);
            assert!(std::env::var_os(KEY).is_none());
        }
        assert!(std::env::var_os(KEY).is_none());
    }
}
