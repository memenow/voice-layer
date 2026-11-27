//! Protocol alignment, subprocess lifecycle, and live smoke tests for the Python worker.

use std::collections::BTreeSet;
use std::path::PathBuf;

use voicelayerd::worker::{WORKER_MODULE, WorkerCallError, WorkerCommand};

fn repo_root() -> PathBuf {
    PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")))
}

fn collect_rust_call_methods(source: &str) -> BTreeSet<String> {
    let mut methods = BTreeSet::new();
    for line in source.lines() {
        if line == "#[cfg(test)]" {
            break;
        }
        let Some(call_pos) = line.find(".call") else {
            continue;
        };
        let after_call = &line[call_pos + ".call".len()..];
        let Some(first_quote) = after_call.find('"') else {
            continue;
        };
        let inside = &after_call[first_quote + 1..];
        let Some(close) = inside.find('"') else {
            continue;
        };
        let method = &inside[..close];
        if !method.is_empty() && method.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
            methods.insert(method.to_owned());
        }
    }
    if source.contains("fn initialize(id: String") {
        methods.insert("initialize".to_owned());
    }
    methods
}

fn extract_protocol_doc_methods(contents: &str) -> BTreeSet<String> {
    let mut methods = BTreeSet::new();
    if let Some(start) = contents.find("<h2 id=\"required-methods\"") {
        let after_heading = &contents[start..];
        let body_start = after_heading
            .find("</h2>")
            .map_or(0, |idx| idx + "</h2>".len());
        let section = &after_heading[body_start..];
        let section = section.find("<h2").map_or(section, |end| &section[..end]);
        for literal in voicelayer_testkit::extract_doc_code_literals(section) {
            if !literal.is_empty() && literal.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
                methods.insert(literal);
            }
        }
    }
    methods
}

fn collect_python_dispatched_methods(source: &str) -> BTreeSet<String> {
    let mut methods = BTreeSet::new();
    for line in source.lines() {
        if let Some(idx) = line.find("method == \"") {
            let after = &line[idx + "method == \"".len()..];
            if let Some(close) = after.find('"') {
                let token = &after[..close];
                if !token.is_empty() && token.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
                    methods.insert(token.to_owned());
                }
            }
        }
        if let Some(idx) = line.find("method in {") {
            let after = &line[idx + "method in {".len()..];
            if let Some(close) = after.find('}') {
                let mut search = &after[..close];
                while let Some(start) = search.find('"') {
                    let after_quote = &search[start + 1..];
                    let Some(end) = after_quote.find('"') else {
                        break;
                    };
                    let token = &after_quote[..end];
                    if !token.is_empty()
                        && token.chars().all(|c| c.is_ascii_lowercase() || c == '_')
                    {
                        methods.insert(token.to_owned());
                    }
                    search = &after_quote[end + 1..];
                }
            }
        }
    }
    methods
}

fn shell_worker(response_script: &str) -> WorkerCommand {
    let combined = format!("head -n 1 > /dev/null; {response_script}");
    WorkerCommand::new(
        "sh".to_owned(),
        vec!["-c".to_owned(), combined],
        std::env::current_dir().expect("current_dir"),
    )
}

#[test]
fn worker_module_constant_resolves_to_python_entrypoint() {
    let module_rel = WORKER_MODULE.replace('.', "/");
    let path = repo_root().join(format!("python/{module_rel}.py"));
    let contents = std::fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "WORKER_MODULE `{WORKER_MODULE}` missing at {}: {err}",
            path.display()
        )
    });
    assert!(
        contents.contains("if __name__ == \"__main__\":"),
        "{} must define an `if __name__ == \"__main__\":` entrypoint",
        path.display()
    );
}

#[test]
fn protocol_doc_rust_client_and_python_worker_methods_match() {
    let root = repo_root();
    let doc_source = std::fs::read_to_string(root.join("docs/architecture/worker-protocol.html"))
        .expect("read worker-protocol.html");
    let python_source = std::fs::read_to_string(root.join("python/voicelayer_worker/worker.py"))
        .expect("read worker.py");
    let rust_mod =
        std::fs::read_to_string(format!("{}/src/worker/mod.rs", env!("CARGO_MANIFEST_DIR")))
            .expect("read src/worker/mod.rs");
    let rust_rpc =
        std::fs::read_to_string(format!("{}/src/worker/rpc.rs", env!("CARGO_MANIFEST_DIR")))
            .expect("read src/worker/rpc.rs");
    let rust_source = format!("{rust_mod}\n{rust_rpc}");

    let doc_methods = extract_protocol_doc_methods(&doc_source);
    let python_methods = collect_python_dispatched_methods(&python_source);
    let rust_methods = collect_rust_call_methods(&rust_source);

    assert!(!doc_methods.is_empty());
    assert_eq!(doc_methods, python_methods);
    assert_eq!(rust_methods, python_methods);
}

#[tokio::test]
async fn worker_command_can_call_health_and_list_providers() {
    let worker = WorkerCommand::discover(repo_root());

    let health = worker.health().await.expect("worker health");
    assert_eq!(health.status, "ok");
    assert_eq!(health.protocol, "2.0");

    let providers = worker.list_providers().await.expect("list providers");
    assert!(providers.providers.iter().any(|p| p.id == "whisper_cpp"));

    let mimo = providers
        .providers
        .iter()
        .find(|p| p.id == "mimo_v2_5_asr")
        .expect("mimo_v2_5_asr advertised");
    assert!(!mimo.default_enabled && mimo.experimental);

    let qwen3 = providers
        .providers
        .iter()
        .find(|p| p.id == "qwen3_asr_1_7b")
        .expect("qwen3_asr_1_7b advertised");
    assert!(!qwen3.default_enabled && qwen3.experimental && qwen3.license == "Apache-2.0");
}

#[tokio::test]
async fn worker_accepts_default_init_payload_and_serves_health() {
    let payload = voicelayer_core::VoiceLayerConfig::default().worker_payload();
    let worker = WorkerCommand::discover(repo_root()).with_init_payload(payload);
    let health = worker
        .health()
        .await
        .expect("worker health with default init payload");
    assert_eq!(health.status, "ok");
    assert_eq!(health.whisper_mode.as_deref(), Some("unconfigured"));
    assert!(!health.mimo_configured);
    assert!(!health.qwen3_asr_configured);
}

#[tokio::test]
async fn segment_probe_returns_provider_unavailable_when_vad_not_configured() {
    if std::env::var("VOICELAYER_WHISPER_VAD_ENABLED").is_ok() {
        return;
    }
    let worker = WorkerCommand::discover(repo_root());
    let request = voicelayer_core::SegmentProbeRequest {
        audio_file: "/tmp/voicelayer-probe-unused.wav".to_owned(),
    };
    let error = worker
        .segment_probe(&request)
        .await
        .expect_err("unconfigured VAD must fail");
    match error {
        WorkerCallError::Rpc(rpc) => {
            assert_eq!(rpc.code, -32004);
            assert!(rpc.message.contains("VOICELAYER_WHISPER_VAD_ENABLED"));
        }
        other => panic!("expected Rpc(-32004), got {other:?}"),
    }
}

#[tokio::test]
async fn worker_call_handles_transport_and_protocol_errors() {
    let missing = WorkerCommand::new(
        "/voicelayer-test/definitely-not-a-real-binary".to_owned(),
        vec![],
        std::env::current_dir().expect("current_dir"),
    );
    assert!(matches!(
        missing.health().await,
        Err(WorkerCallError::Io(_))
    ));

    assert!(matches!(
        shell_worker("exit 0").health().await,
        Err(WorkerCallError::EmptyResponse)
    ));
    assert!(matches!(
        shell_worker("printf '%s\\n' 'not-json'").health().await,
        Err(WorkerCallError::Json(_))
    ));
    assert!(matches!(
        shell_worker(r#"printf '%s\n' '{"jsonrpc":"1.0","id":null,"result":{"providers":[]}}'"#)
            .list_providers()
            .await,
        Err(WorkerCallError::InvalidProtocolVersion(version)) if version == "1.0"
    ));
    assert!(matches!(
        shell_worker(r#"printf '%s\n' '{"jsonrpc":"2.0","id":null}'"#)
            .list_providers()
            .await,
        Err(WorkerCallError::MalformedResponse)
    ));
    assert!(matches!(
        shell_worker(
            r#"printf '%s\n' '{"jsonrpc":"2.0","id":null,"result":{"providers":[]}}'; exit 7"#
        )
        .list_providers()
        .await,
        Err(WorkerCallError::ProcessExited(Some(7), _))
    ));
    assert!(matches!(
        shell_worker(
            r#"printf '%s\n' '{"jsonrpc":"2.0","id":null,"error":{"code":-32004,"message":"provider unavailable"}}'"#
        )
        .list_providers()
        .await,
        Err(WorkerCallError::Rpc(rpc)) if rpc.code == -32004
    ));
}

#[tokio::test]
async fn persistent_worker_reuses_live_child_and_respawns_after_exit() {
    let tempdir = tempfile::tempdir().expect("tempdir");

    let reuse_counter = tempdir.path().join("reuse_counter");
    let reuse_script = format!(
        r#"echo x >> {}; while IFS= read -r _; do printf '%s\n' '{{"jsonrpc":"2.0","id":null,"result":{{"providers":[]}}}}'; done"#,
        reuse_counter.display()
    );
    let persistent = WorkerCommand::new(
        "sh".to_owned(),
        vec!["-c".to_owned(), reuse_script],
        std::env::current_dir().expect("current_dir"),
    );
    persistent.list_providers().await.expect("first call");
    persistent.list_providers().await.expect("second call");
    assert_eq!(
        std::fs::read_to_string(&reuse_counter)
            .expect("counter")
            .lines()
            .count(),
        1
    );

    let respawn_counter = tempdir.path().join("respawn_counter");
    let respawn_script = format!(
        r#"echo x >> {}; head -n 1 > /dev/null; printf '%s\n' '{{"jsonrpc":"2.0","id":null,"result":{{"providers":[]}}}}'"#,
        respawn_counter.display()
    );
    let fragile = WorkerCommand::new(
        "sh".to_owned(),
        vec!["-c".to_owned(), respawn_script],
        std::env::current_dir().expect("current_dir"),
    );
    fragile.list_providers().await.expect("first call");
    fragile.list_providers().await.expect("second call");
    assert_eq!(
        std::fs::read_to_string(&respawn_counter)
            .expect("counter")
            .lines()
            .count(),
        2
    );
}
