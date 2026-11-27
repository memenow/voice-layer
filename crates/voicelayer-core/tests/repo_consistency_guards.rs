//! Cross-repository manifest, toolchain, CI, systemd, and installer consistency guards.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")))
}

fn read_repo_file(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
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

fn extract_yaml_string_value(contents: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}: \"");
    for line in contents.lines() {
        if let Some(rest) = line.trim_start().strip_prefix(&prefix)
            && let Some(end) = rest.find('"')
        {
            return Some(rest[..end].to_owned());
        }
    }
    None
}

fn extract_yaml_bare_string_value(contents: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}: ");
    for line in contents.lines() {
        let Some(rest) = line.trim_start().strip_prefix(&prefix) else {
            continue;
        };
        let stripped = rest.trim_start_matches(['"', '\'']);
        let token: String = stripped
            .chars()
            .take_while(|c| !c.is_whitespace() && *c != '"' && *c != '\'')
            .collect();
        if !token.is_empty() {
            return Some(token);
        }
    }
    None
}

fn extract_python_version_floor(contents: &str) -> Option<String> {
    let value = extract_toml_string_value(contents, "requires-python")?;
    let trimmed = value.trim();
    let stripped = trimmed
        .strip_prefix(">=")
        .or_else(|| trimmed.strip_prefix('>'))
        .unwrap_or(trimmed)
        .trim();
    (!stripped.is_empty()).then(|| stripped.to_owned())
}

fn extract_uv_python_install_version(contents: &str) -> Option<String> {
    let needle = "uv python install ";
    for line in contents.lines() {
        if let Some(idx) = line.find(needle) {
            let token: String = line[idx + needle.len()..]
                .chars()
                .take_while(|c| !c.is_whitespace())
                .collect();
            if !token.is_empty() {
                return Some(token);
            }
        }
    }
    None
}

fn extract_pyproject_license_text(contents: &str) -> Option<String> {
    for line in contents.lines() {
        if !line.trim_start().starts_with("license = ") {
            continue;
        }
        let needle = "text = \"";
        if let Some(idx) = line.find(needle) {
            let after = &line[idx + needle.len()..];
            if let Some(end) = after.find('"') {
                return Some(after[..end].to_owned());
            }
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

#[derive(Debug, PartialEq, Eq)]
enum CargoDepKind {
    WorkspaceInherited,
    InlineOrLiteral,
}

fn classify_cargo_dep_header(trimmed: &str) -> Option<(String, CargoDepKind)> {
    let no_comment = trimmed.split('#').next()?.trim();
    if no_comment.is_empty() {
        return None;
    }
    let (lhs, rhs) = no_comment.split_once('=')?;
    let rhs = rhs.trim();
    if rhs.is_empty() {
        return None;
    }
    let lhs = lhs.trim();
    let (name, suffix) = lhs
        .split_once('.')
        .map_or((lhs, ""), |(n, s)| (n, s.trim()));
    if name.is_empty()
        || !name.starts_with(|c: char| c.is_ascii_alphabetic())
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return None;
    }
    let uses_workspace = suffix == "workspace" || rhs.contains("workspace = true");
    let kind = if uses_workspace {
        CargoDepKind::WorkspaceInherited
    } else {
        CargoDepKind::InlineOrLiteral
    };
    Some((name.to_owned(), kind))
}

fn extract_workspace_dependency_names(contents: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut in_block = false;
    let mut depth: i32 = 0;
    for line in contents.lines() {
        let trimmed = line.trim();
        if depth == 0 && trimmed.starts_with('[') {
            in_block = trimmed.starts_with("[workspace.dependencies]");
            continue;
        }
        if in_block
            && depth == 0
            && let Some((name, _)) = classify_cargo_dep_header(trimmed)
        {
            names.insert(name);
        }
        depth =
            (depth + line.matches('{').count() as i32 - line.matches('}').count() as i32).max(0);
    }
    names
}

fn classify_crate_dependency_forms(contents: &str) -> (BTreeSet<String>, BTreeSet<String>) {
    let mut inherited = BTreeSet::new();
    let mut inline_or_literal = BTreeSet::new();
    let mut in_dep_block = false;
    let mut depth: i32 = 0;
    for line in contents.lines() {
        let trimmed = line.trim();
        if depth == 0 && trimmed.starts_with('[') {
            in_dep_block = trimmed.starts_with("[dependencies]")
                || trimmed.starts_with("[dev-dependencies]")
                || trimmed.starts_with("[build-dependencies]")
                || (trimmed.starts_with("[target.") && trimmed.ends_with(".dependencies]"));
            continue;
        }
        if in_dep_block
            && depth == 0
            && let Some((name, kind)) = classify_cargo_dep_header(trimmed)
        {
            match kind {
                CargoDepKind::WorkspaceInherited => {
                    inherited.insert(name);
                }
                CargoDepKind::InlineOrLiteral => {
                    inline_or_literal.insert(name);
                }
            }
        }
        depth =
            (depth + line.matches('{').count() as i32 - line.matches('}').count() as i32).max(0);
    }
    (inherited, inline_or_literal)
}

fn extract_pep508_names_from_inline(line: &str, out: &mut BTreeSet<String>) {
    let mut search = line;
    while let Some(idx) = search.find('"') {
        let after = &search[idx + 1..];
        let Some(close) = after.find('"') else {
            break;
        };
        let spec = &after[..close];
        search = &after[close + 1..];
        let stop = ['<', '>', '=', '!', '~', '[', '(', ' ', ';'];
        let end = spec.find(stop).unwrap_or(spec.len());
        let name = spec[..end].trim();
        if !name.is_empty() {
            out.insert(name.to_owned());
        }
    }
}

fn extract_pyproject_array(contents: &str, section: &str, key: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut in_section = false;
    let mut in_array = false;
    let array_open = format!("{key} = [");
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_section = trimmed.starts_with(section);
            in_array = false;
            continue;
        }
        if !in_section {
            continue;
        }
        if !in_array {
            if let Some(rest) = trimmed.strip_prefix(&array_open) {
                in_array = true;
                extract_pep508_names_from_inline(rest, &mut names);
                if rest.contains(']') {
                    in_array = false;
                }
            }
            continue;
        }
        if trimmed.starts_with(']') {
            in_array = false;
            continue;
        }
        extract_pep508_names_from_inline(trimmed, &mut names);
    }
    names
}

fn collect_python_imports(start: &Path, out: &mut BTreeSet<String>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(start)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_python_imports(&path, out)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("py") {
            let source = std::fs::read_to_string(&path)?;
            for line in source.lines() {
                let trimmed = line.trim_start();
                if trimmed.starts_with('#') {
                    continue;
                }
                let no_comment = trimmed.split('#').next().unwrap_or(trimmed).trim_end();
                if let Some(rest) = no_comment.strip_prefix("import ") {
                    for part in rest.split(',') {
                        let pkg = part.split_whitespace().next().unwrap_or("");
                        let root = pkg.split('.').next().unwrap_or(pkg);
                        if !root.is_empty() {
                            out.insert(root.to_owned());
                        }
                    }
                } else if let Some(rest) = no_comment.strip_prefix("from ") {
                    let pkg = rest.split_whitespace().next().unwrap_or("");
                    let root = pkg.split('.').next().unwrap_or(pkg);
                    if !root.is_empty() {
                        out.insert(root.to_owned());
                    }
                }
            }
        }
    }
    Ok(())
}

fn extract_bash_chain(contents: &str) -> Vec<String> {
    let mut commands = Vec::new();
    let mut in_block = false;
    for line in contents.lines() {
        let trimmed = line.trim();
        if !in_block {
            if trimmed.starts_with("```bash") {
                in_block = true;
            }
            continue;
        }
        if trimmed.starts_with("```") {
            break;
        }
        let cleaned = trimmed
            .trim_end_matches('\\')
            .trim()
            .trim_start_matches("&& ")
            .trim()
            .to_owned();
        if !cleaned.is_empty() {
            commands.push(cleaned);
        }
    }
    commands
}

fn extract_ci_verify_commands(contents: &str) -> Vec<String> {
    let mut commands = Vec::new();
    for line in contents.lines() {
        let Some(rest) = line.trim_start().strip_prefix("run: ") else {
            continue;
        };
        let cmd = rest.trim();
        if cmd.starts_with("cargo fmt ")
            || cmd.starts_with("cargo clippy ")
            || cmd.starts_with("cargo test ")
            || cmd == "cargo test"
            || cmd.starts_with("uv run ")
        {
            commands.push(cmd.to_owned());
        }
    }
    commands
}

fn normalize_fmt_command(cmd: &str) -> String {
    if let Some(stripped) = cmd.strip_suffix(" -- --check")
        && stripped.starts_with("cargo fmt")
    {
        return stripped.to_owned();
    }
    cmd.to_owned()
}

fn extract_install_sh_bin_targets(source: &str) -> BTreeSet<String> {
    let needle = "\"${BIN_DIR}/";
    let mut targets = BTreeSet::new();
    for line in source.lines() {
        if let Some(idx) = line.find(needle) {
            let after = &line[idx + needle.len()..];
            if let Some(end) = after.find('"')
                && !after[..end].is_empty()
            {
                targets.insert(after[..end].to_owned());
            }
        }
    }
    targets
}

fn extract_install_sh_cargo_build_bins(source: &str) -> BTreeSet<String> {
    let mut bins = BTreeSet::new();
    let mut search = source;
    while let Some(idx) = search.find("--bin ") {
        let after = &search[idx + "--bin ".len()..];
        let end = after
            .find(|c: char| c.is_whitespace() || matches!(c, '\\' | ')' | '"' | '\''))
            .unwrap_or(after.len());
        let name = after[..end].trim();
        if !name.is_empty() {
            bins.insert(name.to_owned());
        }
        search = &after[end..];
    }
    bins
}

fn extract_install_sh_install_pairs(source: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let collapsed = source.replace("\\\n", " ");
    for line in collapsed.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("install -m") {
            continue;
        }
        let mut search = trimmed;
        let mut quoted = Vec::new();
        while let Some(idx) = search.find('"') {
            let after = &search[idx + 1..];
            let Some(close) = after.find('"') else {
                break;
            };
            quoted.push(after[..close].to_owned());
            search = &after[close + 1..];
        }
        if quoted.len() >= 2 {
            pairs.push((quoted[0].clone(), quoted[1].clone()));
        }
    }
    pairs
}

fn extract_install_sh_default_env_dir(source: &str) -> Option<String> {
    for line in source.lines() {
        let Some(rest) = line.trim().strip_prefix("ENV_DIR=\"") else {
            continue;
        };
        let idx = rest.find(":-")?;
        let after = &rest[idx + 2..];
        let end = after.find("}\"")?;
        return Some(after[..end].to_owned());
    }
    None
}

fn extract_systemd_execstart_basenames(source: &str) -> BTreeSet<String> {
    let prefix = "ExecStart=%h/.local/bin/";
    let mut basenames = BTreeSet::new();
    for line in source.lines() {
        if let Some(rest) = line.strip_prefix(prefix) {
            let basename: String = rest.chars().take_while(|c| !c.is_whitespace()).collect();
            if !basename.is_empty() {
                basenames.insert(basename);
            }
        }
    }
    basenames
}

fn extract_systemd_environment_file_paths(source: &str) -> Vec<String> {
    source
        .lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("EnvironmentFile=")?;
            let path = rest.strip_prefix('-').unwrap_or(rest).trim().to_owned();
            (!path.is_empty()).then_some(path)
        })
        .collect()
}

#[test]
fn rust_toolchain_pin_is_consistent_across_repo_files() {
    let channel = extract_toml_string_value(&read_repo_file("rust-toolchain.toml"), "channel")
        .expect("channel");
    let cargo_msrv =
        extract_toml_string_value(&read_repo_file("Cargo.toml"), "rust-version").expect("msrv");
    let ci_toolchain =
        extract_yaml_string_value(&read_repo_file(".github/workflows/ci.yml"), "toolchain")
            .expect("ci toolchain");
    let codeql_toolchain =
        extract_yaml_string_value(&read_repo_file(".github/workflows/codeql.yml"), "toolchain")
            .expect("codeql toolchain");

    assert_eq!(channel, cargo_msrv);
    assert_eq!(channel, ci_toolchain);
    assert_eq!(channel, codeql_toolchain);
}

#[test]
fn python_version_and_ruff_target_are_consistent() {
    let pyproject = read_repo_file("pyproject.toml");
    let ci_yml = read_repo_file(".github/workflows/ci.yml");

    let floor = extract_python_version_floor(&pyproject).expect("requires-python");
    let ci_install = extract_uv_python_install_version(&ci_yml).expect("uv python install");
    let ruff_target = extract_toml_string_value(&pyproject, "target-version").expect("ruff target");

    assert_eq!(floor, ci_install);
    assert_eq!(ruff_target, format!("py{}", floor.replace('.', "")));
}

#[test]
fn release_version_and_license_are_consistent_across_manifests() {
    let cargo_toml = read_repo_file("Cargo.toml");
    let pyproject = read_repo_file("pyproject.toml");
    let openapi = read_repo_file("openapi/voicelayerd.v1.yaml");
    let license_file = read_repo_file("LICENSE");

    let cargo_version = extract_toml_string_value(&cargo_toml, "version").expect("cargo version");
    let pyproject_version =
        extract_toml_string_value(&pyproject, "version").expect("pyproject version");
    let openapi_version =
        extract_yaml_bare_string_value(&openapi, "version").expect("openapi version");
    assert_eq!(cargo_version, pyproject_version);
    assert_eq!(cargo_version, openapi_version);

    let cargo_license = extract_toml_string_value(&cargo_toml, "license").expect("cargo license");
    let pyproject_license = extract_pyproject_license_text(&pyproject).expect("pyproject license");
    assert_eq!(cargo_license, pyproject_license);
    assert_eq!(cargo_license, "Apache-2.0");
    assert!(license_file.contains("Apache License") && license_file.contains("Version 2.0"));
}

#[test]
fn cargo_workspace_members_and_manifests_are_consistent() {
    let root = repo_root();
    let cargo_toml = read_repo_file("Cargo.toml");
    let members = extract_cargo_workspace_members(&cargo_toml);
    let workspace_deps = extract_workspace_dependency_names(&cargo_toml);
    assert!(!members.is_empty());
    assert!(!workspace_deps.is_empty());

    let on_disk: BTreeSet<String> = std::fs::read_dir(root.join("crates"))
        .expect("read crates/")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            entry.file_type().ok()?.is_dir().then_some(())?;
            let name = entry.file_name().into_string().ok()?;
            (!name.starts_with('.') && name != "target").then(|| format!("crates/{name}"))
        })
        .collect();
    assert_eq!(members, on_disk);

    for member in &members {
        let manifest = read_repo_file(&format!("{member}/Cargo.toml"));
        let declared_name = extract_toml_string_value(&manifest, "name").expect("package name");
        let dir_name = Path::new(member)
            .file_name()
            .and_then(|s| s.to_str())
            .expect("dir basename");
        assert_eq!(declared_name, dir_name, "{member} manifest name mismatch");
        assert!(
            manifest.contains("version.workspace = true"),
            "{member} must inherit workspace version"
        );

        let (inherited, literal) = classify_crate_dependency_forms(&manifest);
        let violations: Vec<_> = literal
            .intersection(&workspace_deps)
            .filter(|name| !inherited.contains(*name))
            .collect();
        assert!(
            violations.is_empty(),
            "{member} uses non-workspace declarations for workspace deps: {violations:?}"
        );
    }
}

#[test]
fn verification_chain_matches_across_readme_claude_md_and_ci() {
    let readme_chain: Vec<String> = extract_bash_chain(&read_repo_file("README.md"))
        .into_iter()
        .filter(|cmd| !cmd.starts_with("uv sync"))
        .collect();
    let claude_chain = extract_bash_chain(&read_repo_file("CLAUDE.md"));
    let ci_chain: Vec<String> =
        extract_ci_verify_commands(&read_repo_file(".github/workflows/ci.yml"))
            .into_iter()
            .map(|cmd| normalize_fmt_command(&cmd))
            .collect();

    assert!(!claude_chain.is_empty());
    assert_eq!(readme_chain, claude_chain);
    assert_eq!(claude_chain, ci_chain);
}

#[test]
fn pyproject_dev_and_vad_dependencies_are_wired() {
    let pyproject = read_repo_file("pyproject.toml");
    let ci_yml = read_repo_file(".github/workflows/ci.yml");

    let dev_deps = extract_pyproject_array(&pyproject, "[dependency-groups]", "dev");
    let invoked: BTreeSet<String> = extract_ci_verify_commands(&ci_yml)
        .iter()
        .filter_map(|cmd| cmd.strip_prefix("uv run "))
        .filter_map(|tail| tail.split_whitespace().next().map(str::to_owned))
        .collect();
    let unused_dev: Vec<_> = dev_deps.difference(&invoked).collect();
    assert!(unused_dev.is_empty(), "unused dev deps: {unused_dev:?}");

    let vad_deps = extract_pyproject_array(&pyproject, "[project.optional-dependencies]", "vad");
    assert!(!vad_deps.is_empty());
    let mut imports = BTreeSet::new();
    collect_python_imports(&repo_root().join("python/voicelayer_worker"), &mut imports)
        .expect("walk python/voicelayer_worker");
    let missing_vad: Vec<_> = vad_deps.difference(&imports).collect();
    assert!(
        missing_vad.is_empty(),
        "unimported vad deps: {missing_vad:?}"
    );
}

#[test]
fn systemd_units_and_install_script_stay_aligned() {
    let root = repo_root();
    let install_sh = read_repo_file("scripts/install.sh");
    let bin_targets = extract_install_sh_bin_targets(&install_sh);
    let built_bins = extract_install_sh_cargo_build_bins(&install_sh);
    let install_pairs = extract_install_sh_install_pairs(&install_sh);
    let env_dir_default =
        extract_install_sh_default_env_dir(&install_sh).expect("install.sh ENV_DIR default");
    let install_env_suffix = format!(
        "{}/voicelayerd.env",
        env_dir_default
            .strip_prefix("${HOME}/")
            .unwrap_or(&env_dir_default)
    );

    for (source, dest) in install_pairs {
        if !dest.contains("${BIN_DIR}") {
            continue;
        }
        if let Some(name) = source.strip_prefix("${REPO_ROOT}/target/release/") {
            assert!(
                built_bins.contains(name),
                "install.sh copies unbuilt binary `{name}`"
            );
        } else if let Some(rel) = source.strip_prefix("${REPO_ROOT}/") {
            assert!(
                root.join(rel).exists(),
                "install.sh references missing source `{rel}`"
            );
        }
    }

    for entry in std::fs::read_dir(root.join("systemd")).expect("read systemd/") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|s| s.to_str()) != Some("service") {
            continue;
        }
        let unit = std::fs::read_to_string(&path).expect("read unit");
        for basename in extract_systemd_execstart_basenames(&unit) {
            assert!(
                bin_targets.contains(&basename),
                "{} ExecStart `{basename}` not installed by scripts/install.sh",
                path.display()
            );
        }
        let env_paths = extract_systemd_environment_file_paths(&unit);
        assert!(
            !env_paths.is_empty(),
            "{} missing EnvironmentFile=",
            path.display()
        );
        for env_path in env_paths {
            let suffix = env_path.strip_prefix("%h/").unwrap_or(&env_path);
            assert_eq!(suffix, install_env_suffix);
        }
    }
}
