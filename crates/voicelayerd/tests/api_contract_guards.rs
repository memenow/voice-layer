//! Route, OpenAPI endpoint, and SSE event alignment guards for `voicelayerd`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

const HTTP_METHODS: &[&str] = &[
    "get", "post", "put", "delete", "patch", "head", "options", "trace",
];

fn repo_root() -> PathBuf {
    PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")))
}

fn collect_axum_route_methods(source: &str) -> BTreeMap<String, BTreeSet<String>> {
    let mut routes: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for line in source.lines() {
        if line.trim() == "#[cfg(test)]" {
            break;
        }
        let Some(after_open) = line.trim_start().strip_prefix(".route(\"") else {
            continue;
        };
        let Some(quote_end) = after_open.find('"') else {
            continue;
        };
        let path = after_open[..quote_end].to_owned();
        let tail = after_open[quote_end + 1..]
            .trim_start_matches(',')
            .trim_start();
        let Some(open_paren) = tail.find('(') else {
            continue;
        };
        let method = tail[..open_paren].trim().to_owned();
        if HTTP_METHODS.contains(&method.as_str()) {
            routes.entry(path).or_default().insert(method);
        }
    }
    routes
}

fn collect_openapi_path_methods(contents: &str) -> BTreeMap<String, BTreeSet<String>> {
    let mut paths: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut in_paths = false;
    let mut current_path: Option<String> = None;

    for line in contents.lines() {
        if line == "paths:" {
            in_paths = true;
            continue;
        }
        if in_paths && !line.is_empty() && line.chars().next().is_some_and(|c| !c.is_whitespace()) {
            in_paths = false;
            current_path = None;
            continue;
        }
        if !in_paths {
            continue;
        }
        if let Some(rest) = line.strip_prefix("  /")
            && let Some(name) = rest.strip_suffix(':')
            && !name.contains(' ')
            && !name.is_empty()
        {
            let path = format!("/{name}");
            paths.entry(path.clone()).or_default();
            current_path = Some(path);
            continue;
        }
        if let Some(path) = &current_path
            && let Some(rest) = line.strip_prefix("    ")
            && !rest.starts_with(' ')
            && let Some(colon) = rest.find(':')
        {
            let method = &rest[..colon];
            if HTTP_METHODS.contains(&method) {
                paths
                    .get_mut(path)
                    .expect("current path exists")
                    .insert(method.to_owned());
            }
        }
    }
    paths
}

fn extract_doc_v1_endpoint_paths(contents: &str) -> BTreeSet<String> {
    let mut paths = BTreeSet::new();
    let mut search = contents;
    while let Some(idx) = search.find("/v1/") {
        let after = &search[idx + 1..];
        let token: String = after
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '/'))
            .collect();
        search = &after[token.len()..];
        if token.len() <= "v1/".len() {
            continue;
        }
        let cleaned = token.trim_end_matches('/');
        if !cleaned.is_empty() {
            paths.insert(format!("/{cleaned}"));
        }
    }
    paths
}

fn extract_llm_external_endpoints(source: &str) -> BTreeSet<String> {
    let mut paths = BTreeSet::new();
    for line in source.lines() {
        if line.trim_start().starts_with('#') {
            continue;
        }
        let mut search = line;
        while let Some(idx) = search.find("/v1/") {
            let after = &search[idx + 4..];
            let suffix: String = after
                .chars()
                .take_while(|c| {
                    c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '-' | '_' | '/')
                })
                .collect();
            search = &after[suffix.len()..];
            if suffix
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_lowercase())
            {
                let cleaned = suffix.trim_end_matches('/');
                if !cleaned.is_empty() {
                    paths.insert(format!("/v1/{cleaned}"));
                }
            }
        }
    }
    paths
}

fn collect_event_names(source: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut search = source;
    while let Some(idx) = search.find("=> \"") {
        let after = &search[idx + 4..];
        search = after;
        if let Some(end) = after.find('"')
            && !after[..end].is_empty()
        {
            names.insert(after[..end].to_owned());
        }
    }
    names
}

fn collect_doc_dictation_events(contents: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut search = contents;
    while let Some(idx) = search.find("dictation.") {
        let after = &search[idx..];
        let token: String = after
            .chars()
            .take_while(|c| c.is_ascii_lowercase() || *c == '_' || *c == '.')
            .collect();
        search = &after[token.len().max(1)..];
        if let Some(rest) = token.strip_prefix("dictation.")
            && !rest.is_empty()
            && rest.chars().all(|c| c.is_ascii_lowercase() || c == '_')
        {
            names.insert(format!("dictation_{rest}"));
        }
    }
    names
}

#[test]
fn every_axum_route_method_matches_openapi_bidirectionally() {
    let source = std::fs::read_to_string(format!("{}/src/api/mod.rs", env!("CARGO_MANIFEST_DIR")))
        .expect("read src/api/mod.rs");
    let openapi = std::fs::read_to_string(repo_root().join("openapi/voicelayerd.v1.yaml"))
        .expect("read openapi/voicelayerd.v1.yaml");

    let routes = collect_axum_route_methods(&source);
    let paths = collect_openapi_path_methods(&openapi);
    assert!(!routes.is_empty());
    assert_eq!(routes, paths);
}

#[test]
fn every_doc_v1_endpoint_resolves_to_daemon_route_or_llm_endpoint() {
    let api_source =
        std::fs::read_to_string(format!("{}/src/api/mod.rs", env!("CARGO_MANIFEST_DIR")))
            .expect("read src/api/mod.rs");
    let mut allowed: BTreeSet<String> = collect_axum_route_methods(&api_source)
        .into_keys()
        .collect();

    let root = repo_root();
    let llm_source = std::fs::read_to_string(
        root.join("python/voicelayer_worker/providers/openai_compatible.py"),
    )
    .expect("read openai_compatible.py");
    let llm_endpoints = extract_llm_external_endpoints(&llm_source);
    assert!(!llm_endpoints.is_empty());
    allowed.extend(llm_endpoints);

    let mut docs = vec![root.join("README.md")];
    voicelayer_testkit::collect_html_doc_files(&root.join("docs"), &mut docs).expect("walk docs/");

    let mut violations = Vec::new();
    for doc in docs {
        let contents = std::fs::read_to_string(&doc).expect("read doc");
        for mention in extract_doc_v1_endpoint_paths(&contents) {
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
        "unknown /v1 endpoint references in docs:\n  - {}",
        violations.join("\n  - ")
    );
}

#[test]
fn every_doc_dictation_event_resolves_to_a_daemon_event() {
    let root = repo_root();
    let events_source =
        std::fs::read_to_string(root.join("crates/voicelayer-core/src/domain/events.rs"))
            .expect("read domain/events.rs");
    let emitted = collect_event_names(&events_source);
    assert!(!emitted.is_empty());

    let mut total_events = 0usize;
    let mut unknown = Vec::new();
    for rel in ["docs/architecture/overview.html", "README.md"] {
        let contents = std::fs::read_to_string(root.join(rel)).expect("read doc");
        let mentioned = collect_doc_dictation_events(&contents);
        total_events += mentioned.len();
        for event in mentioned.difference(&emitted) {
            let stripped = event.strip_prefix("dictation_").unwrap_or(event.as_str());
            if !emitted.contains(stripped) {
                unknown.push(format!("{rel}: `{event}`"));
            }
        }
    }
    assert!(total_events > 0);
    assert!(
        unknown.is_empty(),
        "unknown dictation events in docs:\n  - {}",
        unknown.join("\n  - ")
    );
}

#[tokio::test]
async fn transcriptions_route_validates_provider_id_and_translation_before_worker_call() {
    use std::collections::HashMap;
    use std::sync::Arc;

    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use tokio::sync::{Mutex, RwLock};
    use tower::ServiceExt;
    use voicelayerd::{
        DaemonConfig,
        api::{AppState, router},
        events::EventBus,
        session::SessionStore,
    };

    let state = AppState {
        sessions: SessionStore::new(),
        active: Arc::new(Mutex::new(HashMap::new())),
        events: EventBus::new(),
        health: Arc::new(RwLock::new(None)),
        config: Arc::new(DaemonConfig::with_settings(
            Some(std::env::temp_dir().join("voicelayer-api-test.sock")),
            Some(repo_root()),
            voicelayer_core::VoiceLayerConfig::default(),
        )),
        test_audio_silence: None,
    };
    let app = router(state);

    for payload in [
        r#"{"audio_file":"/tmp/a.wav","provider_id":"nonexistent_asr"}"#,
        r#"{"audio_file":"/tmp/a.wav","provider_id":"qwen3_asr_1_7b","translate_to_english":true}"#,
    ] {
        let req = Request::builder()
            .method("POST")
            .uri("/v1/transcriptions")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .expect("build request");
        let resp = app.clone().oneshot(req).await.expect("oneshot");
        let status = resp.status();
        let body = resp.into_body().collect().await.expect("body").to_bytes();
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "expected 400 for {payload}, got {status}: {}",
            String::from_utf8_lossy(&body)
        );
    }
}
