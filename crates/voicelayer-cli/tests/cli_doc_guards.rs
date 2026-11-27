//! CLI subcommand, flag, and config key alignment guards against repository docs.

use std::collections::BTreeSet;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")))
}

fn pascal_to_kebab(value: &str) -> String {
    let mut out = String::new();
    for (index, ch) in value.chars().enumerate() {
        if ch.is_ascii_uppercase() && index > 0 {
            out.push('-');
        }
        out.push(ch.to_ascii_lowercase());
    }
    out
}

fn extract_readme_vl_subcommands(contents: &str) -> BTreeSet<String> {
    let mut subs = BTreeSet::new();
    for prefix in ["-p voicelayer-cli -- ", "--bin vl -- ", "-p vl -- ", "`vl "] {
        let mut search = contents;
        while let Some(idx) = search.find(prefix) {
            let after = &search[idx + prefix.len()..];
            let token: String = after
                .chars()
                .take_while(|c| c.is_ascii_lowercase() || *c == '-' || c.is_ascii_digit())
                .collect();
            if !token.is_empty() && !token.starts_with('-') {
                subs.insert(token);
            }
            search = after;
        }
    }
    subs
}

fn extract_clap_command_names(source: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut in_enum = false;
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed == "pub enum Command {"
            || trimmed == "pub(crate) enum Command {"
            || trimmed == "enum Command {"
        {
            in_enum = true;
            continue;
        }
        if !in_enum {
            continue;
        }
        if trimmed == "}" {
            break;
        }
        let Some(rest) = line.strip_prefix("    ") else {
            continue;
        };
        if rest.starts_with(' ') {
            continue;
        }
        let token: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if !token.is_empty() && token.starts_with(|c: char| c.is_ascii_uppercase()) {
            names.insert(pascal_to_kebab(&token));
        }
    }
    names
}

fn extract_readme_vl_flags(contents: &str) -> BTreeSet<String> {
    let mut collapsed = String::new();
    for line in contents.lines() {
        let trimmed_end = line.trim_end_matches([' ', '\t']);
        if let Some(stripped) = trimmed_end.strip_suffix('\\') {
            collapsed.push_str(stripped);
            collapsed.push(' ');
        } else {
            collapsed.push_str(trimmed_end);
            collapsed.push('\n');
        }
    }

    let mut flags = BTreeSet::new();
    for line in collapsed.lines() {
        let tail = [
            "cargo run -p voicelayer-cli --",
            "cargo run --bin vl --",
            "cargo run -p vl --",
        ]
        .iter()
        .find_map(|prefix| line.find(prefix).map(|idx| &line[idx + prefix.len()..]));
        let Some(mut search) = tail else {
            continue;
        };
        while let Some(idx) = search.find(" --") {
            let after = &search[idx + 3..];
            let token: String = after
                .chars()
                .take_while(|c| c.is_ascii_lowercase() || *c == '-' || c.is_ascii_digit())
                .collect();
            if !token.is_empty() && !token.starts_with('-') {
                flags.insert(token);
            }
            search = after;
        }
    }
    flags
}

fn has_word_long(text: &str) -> bool {
    text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .any(|word| word == "long")
}

fn parse_explicit_long_name(text: &str) -> Option<String> {
    let idx = text.find("long = \"")?;
    let after = &text[idx + "long = \"".len()..];
    let end = after.find('"')?;
    Some(after[..end].to_owned())
}

fn extract_clap_long_flags(source: &str) -> BTreeSet<String> {
    let mut flags = BTreeSet::new();
    let mut in_arg_attr = false;
    let mut long_seen = false;
    let mut explicit_name: Option<String> = None;

    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("#[arg(") {
            in_arg_attr = true;
            long_seen = false;
            explicit_name = None;
        }
        if in_arg_attr {
            if has_word_long(trimmed) {
                long_seen = true;
                if let Some(name) = parse_explicit_long_name(trimmed) {
                    explicit_name = Some(name);
                }
            }
            if trimmed.ends_with(")]") {
                in_arg_attr = false;
            }
            continue;
        }
        if !long_seen || trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        if let Some(colon) = trimmed.find(':') {
            let name = trimmed[..colon].trim();
            if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                let flag = explicit_name
                    .take()
                    .unwrap_or_else(|| name.replace('_', "-"));
                flags.insert(flag);
            }
        }
        long_seen = false;
        explicit_name = None;
    }
    flags
}

fn extract_supported_config_keys(source: &str) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    let mut in_array = false;
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.contains("SUPPORTED_CONFIG_KEYS") && trimmed.contains('&') {
            in_array = true;
            continue;
        }
        if !in_array {
            continue;
        }
        if trimmed.starts_with("];") {
            break;
        }
        let mut search = trimmed;
        while let Some(idx) = search.find('"') {
            let after = &search[idx + 1..];
            let Some(close) = after.find('"') else {
                break;
            };
            keys.insert(after[..close].to_owned());
            search = &after[close + 1..];
        }
    }
    keys
}

fn extract_doc_foreground_ptt_keys(contents: &str) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    for literal in voicelayer_testkit::extract_doc_code_literals(contents) {
        for token in literal.split_whitespace() {
            let clean = token.trim_matches(|c: char| {
                matches!(
                    c,
                    '"' | '\'' | '(' | ')' | '[' | ']' | ',' | '.' | ';' | ':'
                )
            });
            if let Some(rest) = clean.strip_prefix("foreground_ptt.")
                && !rest.is_empty()
                && !rest.contains('.')
                && rest
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            {
                keys.insert(clean.to_owned());
            }
        }
    }
    keys
}

#[test]
fn every_readme_vl_subcommand_and_flag_resolves_to_clap_definition() {
    let commands_rs = std::fs::read_to_string(format!(
        "{}/src/commands/mod.rs",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("read src/commands/mod.rs");
    let readme = std::fs::read_to_string(repo_root().join("README.md")).expect("read README.md");

    let mentioned_subs = extract_readme_vl_subcommands(&readme);
    let clap_subs = extract_clap_command_names(&commands_rs);
    assert!(!mentioned_subs.is_empty());
    assert!(!clap_subs.is_empty());
    let invalid_subs: Vec<_> = mentioned_subs.difference(&clap_subs).collect();
    assert!(
        invalid_subs.is_empty(),
        "unknown README vl subcommands: {invalid_subs:?}"
    );

    let mentioned_flags = extract_readme_vl_flags(&readme);
    let clap_flags = extract_clap_long_flags(&commands_rs);
    assert!(!mentioned_flags.is_empty());
    assert!(!clap_flags.is_empty());
    let invalid_flags: Vec<_> = mentioned_flags.difference(&clap_flags).collect();
    assert!(
        invalid_flags.is_empty(),
        "unknown README vl flags: {invalid_flags:?}"
    );
}

#[test]
fn every_doc_foreground_ptt_config_key_is_supported() {
    let config_rs =
        std::fs::read_to_string(format!("{}/src/config.rs", env!("CARGO_MANIFEST_DIR")))
            .expect("read src/config.rs");
    let supported = extract_supported_config_keys(&config_rs);
    assert!(!supported.is_empty());

    let root = repo_root();
    let mut docs = vec![root.join("README.md")];
    voicelayer_testkit::collect_html_doc_files(&root.join("docs"), &mut docs).expect("walk docs/");

    let mut violations = Vec::new();
    for doc in docs {
        let contents = std::fs::read_to_string(&doc).expect("read doc");
        for mention in extract_doc_foreground_ptt_keys(&contents) {
            if !supported.contains(&mention) {
                violations.push(format!(
                    "{}: `{mention}`",
                    doc.strip_prefix(&root).unwrap_or(&doc).display()
                ));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "unsupported foreground_ptt config keys in docs:\n  - {}",
        violations.join("\n  - ")
    );
}
