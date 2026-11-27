//! Documentation link, path, environment variable, shortcut, and domain vocabulary guards.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")))
}

fn read_repo_file(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

fn all_doc_files() -> Vec<PathBuf> {
    let root = repo_root();
    let mut files = vec![root.join("README.md")];
    voicelayer_testkit::collect_html_doc_files(&root.join("docs"), &mut files).expect("walk docs/");
    files
}

fn extract_toml_string_value(contents: &str, key: &str) -> Option<String> {
    let prefix = format!("{key} = \"");
    for line in contents.lines() {
        if let Some(rest) = line.trim_start().strip_prefix(&prefix)
            && let Some(end) = rest.find('"')
        {
            return Some(rest[..end].to_owned());
        }
    }
    None
}

fn extract_cargo_workspace_members(contents: &str) -> BTreeSet<String> {
    let mut members = BTreeSet::new();
    let mut in_block = false;
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("members = [") {
            in_block = true;
            continue;
        }
        if in_block {
            if trimmed == "]" {
                break;
            }
            let unquoted = trimmed
                .trim_start_matches('"')
                .trim_end_matches(',')
                .trim_end_matches('"');
            if !unquoted.is_empty() {
                members.insert(unquoted.to_owned());
            }
        }
    }
    members
}

fn extract_doc_local_links(contents: &str) -> Vec<String> {
    let mut links = Vec::new();
    for (open, close_char) in [("](", ')'), ("href=\"", '"')] {
        let mut search = contents;
        while let Some(idx) = search.find(open) {
            let after = &search[idx + open.len()..];
            let Some(close) = after.find(close_char) else {
                break;
            };
            let raw = &after[..close];
            search = &after[close + 1..];
            let path = raw.split(['#', '?']).next().unwrap_or(raw);
            if !path.is_empty()
                && !path.starts_with("http://")
                && !path.starts_with("https://")
                && !path.starts_with("mailto:")
                && !path.starts_with('#')
                && (path.ends_with(".html") || path.ends_with(".md"))
            {
                links.push(path.to_owned());
            }
        }
    }
    links
}

fn extract_doc_file_path_refs(contents: &str) -> BTreeSet<String> {
    const EXTS: &[&str] = &[
        ".rs", ".py", ".toml", ".sh", ".yaml", ".html", ".md", ".service",
    ];
    let mut paths = BTreeSet::new();
    for literal in voicelayer_testkit::extract_doc_code_literals(contents) {
        for candidate in literal.split_whitespace() {
            let inner = candidate
                .trim_matches(|c: char| {
                    matches!(
                        c,
                        '"' | '\'' | '(' | ')' | '[' | ']' | ',' | '.' | ';' | ':'
                    )
                })
                .split('#')
                .next()
                .unwrap_or("");
            if inner.contains('/')
                && !inner.contains('*')
                && EXTS.iter().any(|ext| inner.ends_with(ext))
                && inner.chars().next().is_some_and(|c| c.is_ascii_lowercase())
            {
                paths.insert(inner.to_owned());
            }
        }
    }
    paths
}

fn extract_doc_voicelayer_env_vars(contents: &str) -> BTreeSet<String> {
    let mut mentions = BTreeSet::new();
    let mut search = contents;
    while let Some(idx) = search.find("VOICELAYER_") {
        let after = &search[idx..];
        let token: String = after
            .chars()
            .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
            .collect();
        search = &after[token.len().max(1)..];
        if let Some(rest) = token.strip_prefix("VOICELAYER_")
            && !rest.is_empty()
            && !rest.ends_with('_')
        {
            mentions.insert(format!("VOICELAYER_{rest}"));
        }
    }
    mentions
}

fn extract_env_example_keys(contents: &str) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    for line in contents.lines() {
        let after_hash = line.trim_start().trim_start_matches('#').trim_start();
        let Some(rest) = after_hash.strip_prefix("VOICELAYER_") else {
            continue;
        };
        let name: String = rest
            .chars()
            .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
            .collect();
        if !name.is_empty() && rest[name.len()..].starts_with('=') {
            keys.insert(format!("VOICELAYER_{name}"));
        }
    }
    keys
}

fn extract_python_jsonrpc_error_codes(dir: &Path) -> BTreeSet<i32> {
    let mut codes = BTreeSet::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current).expect("read python dir") {
            let entry = entry.expect("entry");
            let path = entry.path();
            if entry.file_type().expect("type").is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|s| s.to_str()) == Some("py") {
                let source = std::fs::read_to_string(&path).expect("read py");
                for line in source.lines() {
                    let Some(idx) = line.find("_CODE = ") else {
                        continue;
                    };
                    let token: String = line[idx + "_CODE = ".len()..]
                        .chars()
                        .take_while(|c| c.is_ascii_digit() || *c == '-')
                        .collect();
                    if let Ok(value) = token.parse::<i32>()
                        && (-32700..=-32000).contains(&value)
                    {
                        codes.insert(value);
                    }
                }
            }
        }
    }
    codes
}

fn extract_doc_jsonrpc_error_codes(contents: &str) -> BTreeSet<i32> {
    let mut codes = BTreeSet::new();
    let mut search = contents;
    while let Some(idx) = search.find('-') {
        let after = &search[idx..];
        let token: String = after
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '-')
            .collect();
        if let Ok(value) = token.parse::<i32>()
            && (-32700..=-32000).contains(&value)
        {
            codes.insert(value);
        }
        search = &after[token.len().max(1)..];
    }
    codes
}

fn collect_rust_shortcut_ids(source: &str) -> BTreeSet<String> {
    let mut ids = BTreeSet::new();
    for line in source.lines() {
        let Some(rest) = line.trim_start().strip_prefix("pub const SHORTCUT_") else {
            continue;
        };
        let Some(eq) = rest.find(" = \"") else {
            continue;
        };
        let after = &rest[eq + 4..];
        if let Some(end) = after.find('"') {
            ids.insert(after[..end].to_owned());
        }
    }
    ids
}

fn collect_doc_shortcut_ids(contents: &str) -> BTreeSet<String> {
    let mut ids = BTreeSet::new();
    let mut search = contents;
    while let Some(idx) = search.find("voicelayer.") {
        let after = &search[idx..];
        let token: String = after
            .chars()
            .take_while(|c| c.is_ascii_lowercase() || *c == '_' || *c == '.')
            .collect();
        search = &after[token.len().max(1)..];
        if let Some(rest) = token.strip_prefix("voicelayer.")
            && !rest.is_empty()
            && !rest.contains('.')
        {
            ids.insert(token);
        }
    }
    ids
}

fn extract_readme_requirement_floor(contents: &str, lang: &str) -> Option<String> {
    let prefix = format!("- {lang} ");
    for line in contents.lines() {
        let Some(rest) = line.trim_start().strip_prefix(&prefix) else {
            continue;
        };
        let token: String = rest
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        if !token.is_empty() {
            return Some(token);
        }
    }
    None
}

fn extract_readme_crate_refs(contents: &str) -> BTreeSet<String> {
    let mut refs = BTreeSet::new();
    let mut search = contents;
    while let Some(idx) = search.find('`') {
        let after = &search[idx + 1..];
        let Some(close) = after.find('`') else {
            break;
        };
        let inner = &after[..close];
        search = &after[close + 1..];
        if let Some(rest) = inner.strip_prefix("crates/")
            && !rest.is_empty()
            && !rest.contains('/')
            && rest
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            refs.insert(rest.to_owned());
        }
    }
    refs
}

fn extract_claude_domain_types(contents: &str) -> BTreeSet<String> {
    let mut types = BTreeSet::new();
    let mut in_section = false;
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("## ") {
            in_section = trimmed == "## Domain Vocabulary";
            continue;
        }
        if !in_section {
            continue;
        }
        let mut search = line;
        while let Some(idx) = search.find('`') {
            let after = &search[idx + 1..];
            let Some(close) = after.find('`') else {
                break;
            };
            let inner = &after[..close];
            search = &after[close + 1..];
            if !inner.is_empty()
                && inner.starts_with(|c: char| c.is_ascii_uppercase())
                && inner.chars().all(|c| c.is_ascii_alphanumeric())
            {
                types.insert(inner.to_owned());
            }
        }
    }
    types
}

fn collect_rust_pub_types(dir: &Path, out: &mut BTreeSet<String>) {
    for entry in std::fs::read_dir(dir).expect("read src dir") {
        let entry = entry.expect("entry");
        let path = entry.path();
        if entry.file_type().expect("type").is_dir() {
            collect_rust_pub_types(&path, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            let source = std::fs::read_to_string(&path).expect("read rs");
            for line in source.lines() {
                let trimmed = line.trim_start();
                for prefix in ["pub struct ", "pub enum "] {
                    if let Some(rest) = trimmed.strip_prefix(prefix) {
                        let name: String = rest
                            .chars()
                            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                            .collect();
                        if !name.is_empty() && name.starts_with(|c: char| c.is_ascii_uppercase()) {
                            out.insert(name);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn every_doc_cross_reference_resolves_to_an_existing_file() {
    let root = repo_root();
    let mut broken = Vec::new();
    for doc in all_doc_files() {
        let contents = std::fs::read_to_string(&doc).expect("read doc");
        let parent = doc.parent().unwrap_or_else(|| Path::new(""));
        for link in extract_doc_local_links(&contents) {
            let resolved = parent.join(&link);
            if !resolved.exists() {
                broken.push(format!(
                    "{} -> `{link}`",
                    doc.strip_prefix(&root).unwrap_or(&doc).display()
                ));
            }
        }
    }
    assert!(
        broken.is_empty(),
        "broken doc links:\n  - {}",
        broken.join("\n  - ")
    );
}

#[test]
fn every_doc_file_path_reference_resolves_to_an_existing_file() {
    let root = repo_root();
    let mut broken = Vec::new();
    for doc in all_doc_files() {
        let contents = std::fs::read_to_string(&doc).expect("read doc");
        for path in extract_doc_file_path_refs(&contents) {
            let primary = root.join(&path);
            let worker_shorthand = root.join("python/voicelayer_worker").join(&path);
            if !primary.exists() && !worker_shorthand.exists() {
                broken.push(format!(
                    "{} -> `{path}`",
                    doc.strip_prefix(&root).unwrap_or(&doc).display()
                ));
            }
        }
    }
    assert!(
        broken.is_empty(),
        "broken doc file-path references:\n  - {}",
        broken.join("\n  - ")
    );
}

#[test]
fn jsonrpc_error_codes_in_python_match_the_protocol_doc() {
    let python_codes =
        extract_python_jsonrpc_error_codes(&repo_root().join("python/voicelayer_worker"));
    let doc_codes =
        extract_doc_jsonrpc_error_codes(&read_repo_file("docs/architecture/worker-protocol.html"));
    assert!(!python_codes.is_empty());
    assert_eq!(python_codes, doc_codes);
}

#[test]
fn vl_desktop_shortcut_ids_match_overview_doc() {
    let rust_ids =
        collect_rust_shortcut_ids(&read_repo_file("crates/voicelayer-desktop/src/hotkeys.rs"));
    let doc_ids = collect_doc_shortcut_ids(&read_repo_file("docs/architecture/overview.html"));
    assert!(!rust_ids.is_empty());
    assert_eq!(rust_ids, doc_ids);
}

#[test]
fn readme_requirements_and_crate_mentions_match_manifests() {
    let readme = read_repo_file("README.md");
    let cargo_toml = read_repo_file("Cargo.toml");
    let pyproject = read_repo_file("pyproject.toml");

    let readme_rust = extract_readme_requirement_floor(&readme, "Rust").expect("README Rust floor");
    let readme_python =
        extract_readme_requirement_floor(&readme, "Python").expect("README Python floor");
    let cargo_rust = extract_toml_string_value(&cargo_toml, "rust-version").expect("rust-version");
    let pyproject_python = extract_toml_string_value(&pyproject, "requires-python")
        .expect("requires-python")
        .trim_start_matches(">=")
        .to_owned();
    assert_eq!(readme_rust, cargo_rust);
    assert_eq!(readme_python, pyproject_python);

    let member_names: BTreeSet<String> = extract_cargo_workspace_members(&cargo_toml)
        .into_iter()
        .filter_map(|m| m.strip_prefix("crates/").map(str::to_owned))
        .collect();
    let readme_crates = extract_readme_crate_refs(&readme);
    assert_eq!(member_names, readme_crates);
}

#[test]
fn every_doc_voicelayer_env_var_mention_is_declared_in_env_example_or_allowlisted() {
    const ALLOWLIST: &[&str] = &[
        "VOICELAYER_LOG",
        "VOICELAYER_DAEMON_BIN",
        "VOICELAYER_INSTALL_BIN_DIR",
        "VOICELAYER_INSTALL_ENV_DIR",
        "VOICELAYER_INSTALL_UNIT_DIR",
    ];
    let root = repo_root();
    let env_keys = extract_env_example_keys(&read_repo_file("systemd/voicelayerd.env.example"));
    assert!(!env_keys.is_empty());

    let allowed: BTreeSet<String> = env_keys
        .into_iter()
        .chain(ALLOWLIST.iter().map(|s| (*s).to_owned()))
        .collect();

    let mut html_docs = Vec::new();
    voicelayer_testkit::collect_html_doc_files(&root.join("docs"), &mut html_docs)
        .expect("walk docs/");

    let mut violations = Vec::new();
    for doc in html_docs {
        let contents = std::fs::read_to_string(&doc).expect("read doc");
        for mention in extract_doc_voicelayer_env_vars(&contents) {
            if !allowed.contains(&mention) {
                violations.push(format!(
                    "{}: `{mention}`",
                    doc.strip_prefix(&root).unwrap_or(&doc).display()
                ));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "undeclared VOICELAYER env vars in docs:\n  - {}",
        violations.join("\n  - ")
    );
}

#[test]
fn every_claude_md_domain_vocabulary_type_exists_as_a_voicelayer_core_pub_type() {
    let vocab = extract_claude_domain_types(&read_repo_file("CLAUDE.md"));
    assert!(!vocab.is_empty());

    let mut pub_types = BTreeSet::new();
    collect_rust_pub_types(
        &repo_root().join("crates/voicelayer-core/src"),
        &mut pub_types,
    );
    assert!(!pub_types.is_empty());

    let missing: Vec<_> = vocab.difference(&pub_types).collect();
    assert!(
        missing.is_empty(),
        "CLAUDE.md domain vocabulary missing pub types: {missing:?}"
    );
}
