//! Long-running VoiceLayer daemon. Binds the `/v1` control API to a local
//! Unix domain socket, supervises one persistent Python worker for ASR/LLM
//! calls, owns the in-process audio capture pipeline, and streams typed
//! lifecycle events over SSE.
//!
//! Module map: `api/` (router + handlers + RFC 9457 errors), `dictation/`
//! (session pipelines over the capture buffer), `audio/` (cpal capture),
//! `worker/` (persistent JSON-RPC worker), `session/` (bounded store),
//! `events/` (typed bus), `platform/` (per-OS capabilities).

pub mod api;
pub mod audio;
pub mod dictation;
pub mod events;
pub mod platform;
pub mod session;
pub mod worker;

use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use tokio::net::UnixListener;
use tokio::sync::{Mutex, RwLock};
use tracing::{info, warn};
use voicelayer_core::{VoiceLayerConfig, default_runtime_dir};

use crate::{
    api::{AppState, refresh_health_state},
    events::EventBus,
    session::SessionStore,
    worker::WorkerCommand,
};

pub use worker::{WorkerCallError, WorkerHealthResult, WorkerPreviewPayload};

/// Runtime configuration handed to [`run_daemon`]: where the daemon binds
/// its UDS, which directory anchors the Python worker, the unified
/// settings, and the release version operators see in `GET /v1/health`.
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    pub socket_path: PathBuf,
    pub project_root: PathBuf,
    pub worker_command: WorkerCommand,
    pub settings: VoiceLayerConfig,
    pub version: String,
}

impl DaemonConfig {
    /// Load settings from the config file (with `VOICELAYER_*` overrides);
    /// explicit arguments win over the file.
    pub fn new(socket_path: PathBuf) -> Self {
        let settings = VoiceLayerConfig::load().unwrap_or_else(|error| {
            warn!(%error, "failed to load config file; falling back to defaults");
            VoiceLayerConfig::default()
        });
        Self::with_settings(Some(socket_path), None, settings)
    }

    /// Compatibility constructor used by the CLI: explicit socket and
    /// project root, settings from the config file.
    pub fn with_project_root(socket_path: PathBuf, project_root: PathBuf) -> Self {
        let settings = VoiceLayerConfig::load().unwrap_or_else(|error| {
            warn!(%error, "failed to load config file; falling back to defaults");
            VoiceLayerConfig::default()
        });
        Self::with_settings(Some(socket_path), Some(project_root), settings)
    }

    pub fn with_settings(
        socket_path: Option<PathBuf>,
        project_root: Option<PathBuf>,
        settings: VoiceLayerConfig,
    ) -> Self {
        let socket_path = socket_path.unwrap_or_else(|| settings.daemon.socket_path());
        let project_root = project_root.unwrap_or_else(|| settings.daemon.project_root());
        let worker_command = WorkerCommand::discover(project_root.clone())
            .with_init_payload(settings.worker_payload());
        Self {
            socket_path,
            project_root,
            worker_command,
            settings,
            version: env!("CARGO_PKG_VERSION").to_owned(),
        }
    }
}

use std::env;

pub async fn run_daemon(config: DaemonConfig) -> std::io::Result<()> {
    let socket_dir = config
        .socket_path
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    tokio::fs::create_dir_all(&socket_dir).await?;
    // The runtime dir holds recordings and provider state: owner-only.
    set_owner_only(&socket_dir).await;

    // VoiceLayer recordings live under the runtime dir, never shared /tmp.
    let recordings_dir = default_runtime_dir().join("dictation");
    tokio::fs::create_dir_all(&recordings_dir).await?;
    set_owner_only(&recordings_dir).await;

    if tokio::fs::try_exists(&config.socket_path).await? {
        tokio::fs::remove_file(&config.socket_path).await?;
    }
    let listener = UnixListener::bind(&config.socket_path)?;
    set_owner_only(&config.socket_path).await;

    let worker_timeout = Duration::from_secs(config.settings.daemon.worker_timeout_seconds);
    let _ = worker_timeout; // worker timeouts are per-method; budget lives on WorkerCommand.

    let state = AppState {
        sessions: SessionStore::new(),
        active: Arc::new(Mutex::new(HashMap::new())),
        events: EventBus::new(),
        health: Arc::new(RwLock::new(None)),
        config: Arc::new(config),
        test_audio_silence: None,
    };

    // Keep the health snapshot warm, but never let the refresher be the
    // thing that spawns the worker: refresh only once something else has.
    let refresher_state = state.clone();
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(api::HEALTH_REFRESH_INTERVAL);
        loop {
            ticker.tick().await;
            let warm = refresher_state.health.read().await.is_some();
            if warm || refresher_state.config.worker_command.is_running().await {
                refresh_health_state(&refresher_state).await;
            }
        }
    });

    let socket_path = state.config.socket_path.clone();
    let worker = state.config.worker_command.clone();
    let app = api::router(state);

    info!(socket_path = %socket_path.display(), "starting VoiceLayer daemon");
    let serve_result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await;

    worker.shutdown().await;
    let _ = tokio::fs::remove_file(&socket_path).await;
    serve_result
}

async fn set_owner_only(path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions =
            std::fs::Permissions::from_mode(if path.is_dir() { 0o700 } else { 0o600 });
        if let Err(error) = tokio::fs::set_permissions(path, permissions).await {
            warn!(%error, path = %path.display(), "failed to tighten permissions");
        }
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            warn!(%error, "failed to listen for Ctrl+C");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => warn!(%error, "failed to listen for SIGTERM"),
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("shutdown signal received; stopping daemon");
}

#[cfg(test)]
mod openapi_route_alignment_tests {
    //! Cross-check that every axum route declared in
    //! `build_app_router` has a matching `paths:` entry in
    //! `openapi/voicelayerd.v1.yaml`, and that every openapi path has
    //! a matching axum route. The per-schema drift guards in
    //! `voicelayer-core::domain::tests` (#28 / #33 / #34 / #35 / #36 /
    //! #37) cover the *response shape* contract; this module covers
    //! the *endpoint surface* contract — a shipped-but-undocumented
    //! route or a documented-but-unimplemented path was previously
    //! invisible to every existing guard.

    use std::collections::{BTreeMap, BTreeSet};

    /// Allowlist of HTTP method tokens we recognise on either side of
    /// the contract. Keeps the parsers from mistaking other
    /// four-space-indent keys (e.g. `summary:`, `responses:`,
    /// `requestBody:`, `parameters:`) for methods just because they
    /// happen to be at the same depth, and excludes axum builder
    /// helpers like `with_state` that are never legitimate methods.
    const HTTP_METHODS: &[&str] = &[
        "get", "post", "put", "delete", "patch", "head", "options", "trace",
    ];

    /// Walk a source string and pull out every `(path, method)` pair
    /// declared by `.route("PATH", METHOD(...))` calls. Stops at the
    /// first `#[cfg(test)]` line so synthetic fixtures and helper
    /// strings inside test modules do not pollute the production
    /// route set. Production routes in `voicelayerd::lib.rs` all
    /// land in `build_app_router` ahead of every test module.
    fn collect_axum_route_methods(source: &str) -> BTreeMap<String, BTreeSet<String>> {
        let mut routes: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for line in source.lines() {
            if line.trim() == "#[cfg(test)]" {
                break;
            }
            let trimmed = line.trim_start();
            let Some(after_path_open) = trimmed.strip_prefix(".route(\"") else {
                continue;
            };
            let Some(quote_end) = after_path_open.find('"') else {
                continue;
            };
            let path = after_path_open[..quote_end].to_owned();
            // After the closing `"` of the path, the next non-comma /
            // non-space token is the method builder.
            let tail = after_path_open[quote_end + 1..]
                .trim_start_matches(',')
                .trim_start();
            let Some(open_paren) = tail.find('(') else {
                continue;
            };
            let method = tail[..open_paren].trim().to_owned();
            if !HTTP_METHODS.contains(&method.as_str()) {
                // `.route(path, with_state(...))` and similar
                // non-method builder calls would otherwise pollute
                // the set. We only care about the eight HTTP verbs.
                continue;
            }
            routes.entry(path).or_default().insert(method);
        }
        routes
    }

    /// Pull a method token out of a four-space-indent line. Tolerates
    /// both the block form (`    get:` followed by nested keys on
    /// the next line) and the inline form (`    get: {}`,
    /// `    get: someValue`); both end up with the method name in the
    /// substring before the first `:`. Returns `None` when the line is
    /// not at four-space indent, has no colon, or names a key that is
    /// not on `HTTP_METHODS` (so `summary:`, `responses:`,
    /// `requestBody:`, etc. are filtered out).
    fn extract_method_at_method_indent(line: &str) -> Option<&str> {
        let rest = line.strip_prefix("    ")?;
        if rest.starts_with(' ') {
            return None;
        }
        let colon_pos = rest.find(':')?;
        let method = &rest[..colon_pos];
        if HTTP_METHODS.contains(&method) {
            Some(method)
        } else {
            None
        }
    }

    /// Walk an openapi YAML and pull out every `(path, method)` pair.
    /// Path keys are at two-space indent under the `paths:` block;
    /// each method is a four-space-indent key under its parent path
    /// whose name is one of the allowlisted HTTP verbs. Bigger keys
    /// at the same indent (`summary:`, `responses:`, `requestBody:`)
    /// are filtered out by the allowlist.
    fn collect_openapi_path_methods(contents: &str) -> BTreeMap<String, BTreeSet<String>> {
        let mut paths: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut in_paths = false;
        let mut current_path: Option<String> = None;
        for line in contents.lines() {
            if line == "paths:" {
                in_paths = true;
                continue;
            }
            if in_paths
                && !line.is_empty()
                && line.chars().next().is_some_and(|c| !c.is_whitespace())
            {
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
                && let Some(method) = extract_method_at_method_indent(line)
            {
                paths
                    .get_mut(path)
                    .expect("current_path entry was inserted above")
                    .insert(method.to_owned());
            }
        }
        paths
    }

    #[test]
    fn collect_axum_route_methods_extracts_each_method_token() {
        let source = "\
            Router::new()\n\
                .route(\"/v1/health\", get(get_health))\n\
                .route(\"/v1/sessions\", get(list_sessions))\n\
                .route(\"/v1/sessions/dictation\", post(create_dictation_session))\n\
                .with_state(state)\n\
        ";
        let routes = collect_axum_route_methods(source);
        assert_eq!(routes.len(), 3);
        assert!(routes.get("/v1/health").is_some_and(|m| m.contains("get")));
        assert!(
            routes
                .get("/v1/sessions")
                .is_some_and(|m| m.contains("get"))
        );
        assert!(
            routes
                .get("/v1/sessions/dictation")
                .is_some_and(|m| m.contains("post"))
        );
    }

    #[test]
    fn collect_axum_route_methods_ignores_non_method_builders() {
        // `.with_state(state)` looks like `.method(...)` to a naive
        // parser. The HTTP-method allowlist filters it out so a
        // builder chain entry never lands as a synthetic route.
        let source = "\
            Router::new()\n\
                .route(\"/v1/real\", get(real_handler))\n\
                .with_state(state)\n\
                .layer(some_layer())\n\
        ";
        let routes = collect_axum_route_methods(source);
        assert_eq!(routes.len(), 1);
        assert!(routes.get("/v1/real").is_some_and(|m| m.contains("get")));
    }

    #[test]
    fn collect_axum_route_methods_stops_at_first_cfg_test_line() {
        let source = "\
            Router::new()\n\
                .route(\"/v1/real\", get(handler))\n\
            #[cfg(test)]\n\
            mod tests {\n\
                let _ = router.route(\"/v1/synthetic\", get(handler));\n\
            }\n\
        ";
        let routes = collect_axum_route_methods(source);
        assert!(routes.contains_key("/v1/real"));
        assert!(
            !routes.contains_key("/v1/synthetic"),
            "test-block routes must be ignored; got {routes:?}",
        );
    }

    #[test]
    fn collect_openapi_path_methods_extracts_methods_under_each_path() {
        let yaml = concat!(
            "openapi: 3.1.0\n",
            "info:\n",
            "  title: Local Control API\n",
            "paths:\n",
            "  /v1/health:\n",
            "    get:\n",
            "      operationId: getHealth\n",
            "  /v1/sessions/dictation:\n",
            "    post:\n",
            "      operationId: startLiveDictationSession\n",
            "      requestBody:\n",
            "        required: true\n",
            "components:\n",
            "  schemas:\n",
            "    HealthResponse:\n",
            "      type: object\n",
        );
        let paths = collect_openapi_path_methods(yaml);
        assert_eq!(paths.len(), 2);
        assert!(paths.get("/v1/health").is_some_and(|m| m.contains("get")));
        assert!(
            paths
                .get("/v1/sessions/dictation")
                .is_some_and(|m| m.contains("post"))
        );
    }

    #[test]
    fn collect_openapi_path_methods_ignores_non_method_keys_at_method_indent() {
        // `summary:`, `description:`, `responses:`, `requestBody:`,
        // `parameters:` all live at four-space indent under a path
        // entry. Only the eight HTTP verbs are real method keys; the
        // allowlist must filter the rest out.
        let yaml = concat!(
            "paths:\n",
            "  /v1/foo:\n",
            "    summary: not a method\n",
            "    description: also not a method\n",
            "    get:\n",
            "      operationId: getFoo\n",
            "    requestBody:\n",
            "      content: {}\n",
            "components: {}\n",
        );
        let paths = collect_openapi_path_methods(yaml);
        let methods = paths.get("/v1/foo").expect("/v1/foo declared");
        assert_eq!(methods.len(), 1);
        assert!(methods.contains("get"));
    }

    #[test]
    fn collect_openapi_path_methods_ignores_indented_keys_outside_paths_block() {
        let yaml = concat!(
            "paths:\n",
            "  /v1/real:\n",
            "    get: {}\n",
            "components:\n",
            "  schemas:\n",
            "    Foo:\n",
            "      description: |\n",
            "        See /v1/not-a-path: discussed in the changelog.\n",
        );
        let paths = collect_openapi_path_methods(yaml);
        assert_eq!(paths.len(), 1);
        let methods = paths.get("/v1/real").expect("real path");
        assert!(methods.contains("get"));
    }

    #[test]
    fn every_axum_route_method_matches_an_openapi_method_and_vice_versa() {
        let source_path = format!("{}/src/api/mod.rs", env!("CARGO_MANIFEST_DIR"));
        let openapi_path = format!(
            "{}/../../openapi/voicelayerd.v1.yaml",
            env!("CARGO_MANIFEST_DIR"),
        );

        let source = std::fs::read_to_string(&source_path).expect("read voicelayerd api/mod.rs");
        let openapi = std::fs::read_to_string(&openapi_path).expect("read openapi contract");

        let routes = collect_axum_route_methods(&source);
        let paths = collect_openapi_path_methods(&openapi);

        assert!(
            !routes.is_empty(),
            "expected at least one axum route in build_app_router; \
             collect_axum_route_methods may be misparsing",
        );
        assert!(
            !paths.is_empty(),
            "expected at least one path in openapi/voicelayerd.v1.yaml; \
             collect_openapi_path_methods may be misparsing",
        );

        for (path, methods) in &routes {
            let yaml_methods = paths
                .get(path)
                .unwrap_or_else(|| panic!("axum route `{path}` is missing from openapi `paths:`"));
            for method in methods {
                assert!(
                    yaml_methods.contains(method),
                    "axum route `{method} {path}` has no matching openapi method declaration. \
                     Add `{method}:` under `paths.{path}` (or drop the route).",
                );
            }
        }
        for (path, methods) in &paths {
            let axum_methods = routes
                .get(path)
                .unwrap_or_else(|| panic!("openapi path `{path}` has no matching axum route"));
            for method in methods {
                assert!(
                    axum_methods.contains(method),
                    "openapi `{method} {path}` has no matching `.route(...)` call. \
                     Add `.route(\"{path}\", {method}(handler))` (or drop the openapi entry).",
                );
            }
        }
    }
}

#[cfg(test)]
mod doc_v1_endpoint_alignment_tests {
    //! Cross-check that every `/v1/<path>` URL mentioned in any
    //! operator-facing documentation (the project README plus every
    //! standalone HTML page under `docs/`) resolves to either a real axum route in
    //! `build_app_router` or an external OpenAI-compatible LLM
    //! endpoint the daemon talks *to* (e.g.
    //! `/v1/chat/completions`, `/v1/models`).
    //!
    //! The drift mode is silent and operator-facing: a doc claims
    //! `POST /v1/sessions/dictation/start` is the right entrypoint
    //! after the route was renamed from `/start` to bare
    //! `/sessions/dictation`, the operator copy-pastes the URL into
    //! curl, the daemon returns 404, and the only fix is to read
    //! the actual router source.
    //!
    //! The LLM allowlist is no longer a hard-coded const inside
    //! this test. It is derived at test time from the Python worker
    //! module
    //! `python/voicelayer_orchestrator/providers/llm_openai_compatible.py`,
    //! which is the single source of truth for the OpenAI-compatible
    //! suffixes the daemon dials. If that worker migrates to a new
    //! suffix (e.g. `/v1/responses`), the allowlist tracks
    //! automatically without a parallel edit here.
    //!
    //! Reverse direction (every axum route is mentioned in some
    //! doc) is intentionally not enforced — the openapi document
    //! is the canonical surface contract (see
    //! `openapi_route_alignment_tests`); docs prose is merely a
    //! guide.
    use std::collections::BTreeSet;

    /// Walk a documentation body and pull every `/v1/<path>` URL token.
    /// Anchors on the literal `/v1/` prefix and walks forward
    /// taking ASCII alphanumerics, hyphens, underscores, and
    /// internal `/` separators. Stops at any other character (a
    /// space, punctuation, backtick, etc.) so trailing prose like
    /// `/v1/health.` (sentence-final period) yields the bare path.
    ///
    /// Uppercase letters are intentionally captured (not folded to
    /// lowercase) so that operator-facing typos such as
    /// `/v1/Sessions/Dictation` survive extraction and surface as
    /// allowlist violations downstream, instead of being silently
    /// truncated at the first uppercase character.
    fn extract_doc_v1_endpoint_paths(contents: &str) -> BTreeSet<String> {
        let mut paths = BTreeSet::new();
        let mut search = contents;
        while let Some(idx) = search.find("/v1/") {
            let after = &search[idx + 1..];
            let token: String = after
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '/'))
                .collect();
            // Step past the full captured token so pathological
            // inputs like `/v1/v1/foo` are not double-counted by a
            // re-scan of the same suffix.
            search = &after[token.len()..];
            if token.len() <= "v1/".len() {
                continue;
            }
            // Trim a trailing `/` so `POST /v1/sessions/dictation/`
            // (docs prose form) collapses onto the bare path.
            let cleaned = token.trim_end_matches('/');
            if cleaned.is_empty() {
                continue;
            }
            paths.insert(format!("/{cleaned}"));
        }
        paths
    }

    /// Walk this file's source and pull the first-argument path of
    /// every `.route("PATH", ...)` call. Stops at the first
    /// `#[cfg(test)]` line so synthetic fixtures and helper strings
    /// inside test modules do not pollute the production route set
    /// (the same truncation pattern as
    /// `collect_axum_route_methods` in
    /// `openapi_route_alignment_tests`).
    ///
    /// Assumption: in this file, `#[cfg(test)]` is only used as the
    /// test-module boundary, never to gate individual production
    /// `.route(...)` calls (e.g. a debug-only handler). If that ever
    /// changes, this scanner needs a smarter delimiter, otherwise it
    /// will silently truncate the production route set at the first
    /// such guard.
    fn collect_axum_route_paths(source: &str) -> BTreeSet<String> {
        let mut paths = BTreeSet::new();
        for line in source.lines() {
            if line.trim() == "#[cfg(test)]" {
                break;
            }
            let trimmed = line.trim_start();
            let Some(after) = trimmed.strip_prefix(".route(\"") else {
                continue;
            };
            let Some(end) = after.find('"') else {
                continue;
            };
            paths.insert(after[..end].to_owned());
        }
        paths
    }

    /// Walk the Python OpenAI-compatible worker source and pull
    /// every `/v1/<segment>` literal it mentions. The worker is the
    /// single source of truth for the OpenAI-compatible suffixes the
    /// daemon dials, so this extractor lets the doc-alignment guard
    /// follow the worker without a parallel const edit here.
    ///
    /// Matching rule:
    ///
    /// - Anchor on the literal `/v1/` prefix.
    /// - Walk forward taking lowercase ASCII letters, digits,
    ///   hyphens, underscores, and internal `/` separators (the same
    ///   character set the worker uses for its f-string suffixes).
    /// - Trim a trailing `/` so a stray `/v1/foo/` collapses onto
    ///   the bare path.
    /// - Skip any line whose first non-whitespace character is `#`
    ///   so commented-out URLs (e.g. `# /v1/should-not-be-captured`)
    ///   never reach the allowlist.
    fn extract_llm_external_endpoints_from_python_worker(source: &str) -> BTreeSet<String> {
        let mut paths = BTreeSet::new();
        for line in source.lines() {
            if line.trim_start().starts_with('#') {
                continue;
            }
            let mut search = line;
            while let Some(idx) = search.find("/v1/") {
                // `/v1/` is 4 ASCII bytes, so slicing past the
                // matched prefix is safe.
                let after_v1_slash = &search[idx + 4..];
                let suffix: String = after_v1_slash
                    .chars()
                    .take_while(|c| {
                        c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '-' | '_' | '/')
                    })
                    .collect();
                // Step past the full captured token (`/v1/` plus
                // the suffix we just consumed) so pathological
                // inputs like `/v1/v1/foo` are not double-counted
                // by a re-scan of the same suffix. Mirrors the
                // advancement strategy in
                // `extract_doc_v1_endpoint_paths` above.
                search = &after_v1_slash[suffix.len()..];
                // Require a real first segment that starts with a
                // lowercase ASCII letter, mirroring how the worker
                // names its OpenAI-compatible paths
                // (`chat/completions`, `models`, ...).
                let Some(first_char) = suffix.chars().next() else {
                    continue;
                };
                if !first_char.is_ascii_lowercase() {
                    continue;
                }
                let cleaned = suffix.trim_end_matches('/');
                if cleaned.is_empty() {
                    continue;
                }
                paths.insert(format!("/v1/{cleaned}"));
            }
        }
        paths
    }

    #[test]
    fn extract_llm_external_endpoints_from_python_worker_captures_chat_completions_and_models() {
        let fixture = "\
\"\"\"OpenAI-compatible chat completion provider fixture.\"\"\"


def resolve_chat_completions_url(endpoint: str) -> str:
    normalized = endpoint.rstrip(\"/\")
    if normalized.endswith(\"/v1\"):
        return f\"{normalized}/chat/completions\"
    return f\"{normalized}/v1/chat/completions\"


def resolve_models_url(endpoint: str) -> str:
    normalized = endpoint.rstrip(\"/\")
    if normalized.endswith(\"/v1/chat/completions\"):
        return normalized.removesuffix(\"/chat/completions\") + \"/models\"
    return f\"{normalized}/v1/models\"


# /v1/should-not-be-captured  -- comment line must be filtered out
";
        let paths = extract_llm_external_endpoints_from_python_worker(fixture);
        assert_eq!(
            paths,
            ["/v1/chat/completions", "/v1/models"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            "expected only the two real OpenAI-compatible URLs; the `#`-prefixed \
             comment line `/v1/should-not-be-captured` must be skipped, and the \
             `removesuffix(\"/chat/completions\") + \"/models\"` branch must not \
             produce a spurious capture",
        );
    }

    #[test]
    fn extract_llm_external_endpoints_from_python_worker_does_not_double_count_overlapping_v1_segments()
     {
        let fixture = "return f\"{normalized}/v1/v1/foo\"\n";
        let paths = extract_llm_external_endpoints_from_python_worker(fixture);
        assert_eq!(
            paths,
            ["/v1/v1/foo"].iter().map(|s| (*s).to_owned()).collect(),
            "the loop must step past the full captured token so a \
             pathological worker URL like `/v1/v1/foo` yields one entry, \
             not a second `/v1/foo` re-discovered inside the suffix",
        );
    }

    #[test]
    fn extract_doc_v1_endpoint_paths_handles_method_prefix_and_trailing_punctuation() {
        let md = "\
- `POST /v1/sessions/dictation` starts recording.
- The `GET /v1/events/stream` channel emits SSE.
- See `/v1/health.` for liveness — note the trailing period must drop.
- `/v1/path/with/multiple/segments` should resolve as one entry.
- The string `/v1/` alone (no segment) must not be captured.
";
        let paths = extract_doc_v1_endpoint_paths(md);
        assert_eq!(
            paths,
            [
                "/v1/events/stream",
                "/v1/health",
                "/v1/path/with/multiple/segments",
                "/v1/sessions/dictation",
            ]
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
            "the bare `/v1/` prefix without a path segment must NOT be captured",
        );
    }

    #[test]
    fn extract_doc_v1_endpoint_paths_captures_uppercase_so_doc_typos_surface_as_violations() {
        let md = "- See `/v1/Sessions/Dictation` for the live entry.";
        let paths = extract_doc_v1_endpoint_paths(md);
        assert_eq!(
            paths,
            ["/v1/Sessions/Dictation"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            "uppercase characters must be captured verbatim so an \
             operator-facing typo like `/v1/Sessions/Dictation` \
             reaches the allowlist comparison and surfaces as a \
             violation, instead of being silently truncated to \
             `/v1/` and dropped",
        );
    }

    #[test]
    fn extract_doc_v1_endpoint_paths_does_not_double_count_overlapping_v1_segments() {
        let md = "`/v1/v1/foo` is a degenerate but legal substring.";
        let paths = extract_doc_v1_endpoint_paths(md);
        assert_eq!(
            paths,
            ["/v1/v1/foo"].iter().map(|s| (*s).to_owned()).collect(),
            "the loop must step past the full captured token so a \
             pathological input like `/v1/v1/foo` yields one entry, \
             not a second `/v1/foo` re-discovered inside the suffix",
        );
    }

    #[test]
    fn collect_axum_route_paths_strips_method_builder_and_skips_test_module() {
        let source = "\
fn build_app_router() -> Router {
    Router::new()
        .route(\"/v1/health\", get(get_health))
        .route(\"/v1/sessions\", get(list_sessions))
}

#[cfg(test)]
mod tests {
    let _ = router.route(\"/v1/synthetic\", get(handler));
}
";
        let paths = collect_axum_route_paths(source);
        assert_eq!(
            paths,
            ["/v1/health", "/v1/sessions"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            "the test-module fixture path must be excluded by #[cfg(test)] truncation",
        );
    }

    /// Every `/v1/<path>` URL the README or any doc mentions must
    /// be either a real daemon route or a known external
    /// OpenAI-compatible endpoint. The LLM-side allowlist is
    /// derived at test time from the Python worker module
    /// `python/voicelayer_orchestrator/providers/llm_openai_compatible.py`,
    /// which is the single source of truth for the suffixes the
    /// daemon dials on the configured `VOICELAYER_LLM_ENDPOINT`
    /// host. The same URLs are operator-documented in
    /// `docs/guides/local-llm-provider.html`.
    #[test]
    fn every_doc_v1_endpoint_mention_resolves_to_an_axum_route_or_llm_allowlist() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let lib_source = std::fs::read_to_string(format!("{manifest}/src/api/mod.rs"))
            .expect("read voicelayerd api/mod.rs");
        let routes = collect_axum_route_paths(&lib_source);
        assert!(
            !routes.is_empty(),
            "expected at least one .route() call in voicelayerd lib.rs",
        );

        let repo_root = std::path::PathBuf::from(format!("{manifest}/../.."));
        let python_worker_path =
            repo_root.join("python/voicelayer_orchestrator/providers/llm_openai_compatible.py");
        let python_source = std::fs::read_to_string(&python_worker_path)
            .unwrap_or_else(|err| panic!("read {}: {err}", python_worker_path.display()));
        let llm_external_endpoints =
            extract_llm_external_endpoints_from_python_worker(&python_source);
        assert!(
            !llm_external_endpoints.is_empty(),
            "expected at least one /v1/<path> literal in the python worker — \
             extract_llm_external_endpoints_from_python_worker may be misparsing, \
             or the worker may have moved to a different file path",
        );

        let mut docs = vec![repo_root.join("README.md")];
        voicelayer_doc_test_utils::collect_html_doc_files(&repo_root.join("docs"), &mut docs)
            .expect("walk docs/");

        let mut allowed: BTreeSet<String> = routes;
        allowed.extend(llm_external_endpoints);

        let mut violations: Vec<String> = Vec::new();
        for doc_path in &docs {
            let contents = match std::fs::read_to_string(doc_path) {
                Ok(s) => s,
                Err(_) => continue,
            };
            for mention in extract_doc_v1_endpoint_paths(&contents) {
                if allowed.contains(&mention) {
                    continue;
                }
                violations.push(format!(
                    "{}: `{mention}`",
                    doc_path
                        .strip_prefix(&repo_root)
                        .unwrap_or(doc_path)
                        .display(),
                ));
            }
        }
        assert!(
            violations.is_empty(),
            "docs reference `/v1/...` endpoints that are neither real daemon \
             routes nor on the LLM external-endpoint allowlist:\n  - {}\n\n\
             Either fix the doc URL to match an actual `.route(...)` call in \
             crates/voicelayerd/src/, drop the mention, or add the URL \
             as a `/v1/...` literal in \
             `python/voicelayer_orchestrator/providers/llm_openai_compatible.py`, \
             since that file is the single source of truth scanned by this guard.",
            violations.join("\n  - "),
        );
    }
}

#[cfg(test)]
mod sse_event_doc_alignment_tests {
    //! Cross-check that every code-styled SSE event name in
    //! `docs/architecture/overview.html` corresponds to an
    //! `EventEnvelope::new("<name>", ...)` emission somewhere in
    //! `crates/voicelayerd/src/`. Forward direction only: the
    //! doc enumerates the dictation-pipeline events that operators
    //! consume, but the daemon also emits non-dictation events
    //! (`compose.job_created`, `transcription.completed`, etc.) that
    //! the doc intentionally does not mention. Reverse-direction
    //! enforcement would require lifting the doc to an exhaustive
    //! event reference, which is out of scope for this guard.
    //!
    //! The drift mode the forward direction catches is doc rot: the
    //! doc claims `dictation.fancy_new_event` is emitted but the
    //! daemon never produces it. Operators wiring an SSE consumer
    //! against the doc see no event and have no way to discover the
    //! name was retired.
    use std::collections::BTreeSet;

    /// Walk a Rust source string and pull every event name from
    /// `EventEnvelope::new("<name>", ...)` calls. Tolerates the
    /// multi-line layout the daemon uses today
    /// (`EventEnvelope::new(\n    "<name>",\n ...)`): the scanner
    /// finds each `EventEnvelope::new(` occurrence, skips
    /// whitespace, and reads the next `"..."` literal. Stops at the
    /// first `#[cfg(test)]` line so synthetic test fixtures
    /// (`test.single_event`) do not leak into the production set.
    fn collect_event_envelope_names(source: &str) -> BTreeSet<String> {
        // Events are typed `DaemonEvent` variants; names live in the
        // `DaemonEvent::name()` match arms in voicelayer-core's domain.rs.
        let mut names = BTreeSet::new();
        let mut search = source;
        while let Some(idx) = search.find("=> \"") {
            let after = &search[idx + 4..];
            search = after;
            let Some(end) = after.find('"') else {
                continue;
            };
            let name = &after[..end];
            if !name.is_empty() {
                names.insert(name.to_owned());
            }
        }
        names
    }

    /// Walk documentation prose and pull every token of shape
    /// `dictation.<word>` (lowercase letters and underscores only
    /// after the dot). Designed for the SSE event mentions in
    /// `docs/architecture/overview.html`. Excludes the `worker.*`,
    /// `compose.*`, and `transcription.*` namespaces, which the doc
    /// does not enumerate today and the forward-direction guard does
    /// not require to be enumerated.
    fn collect_doc_dictation_event_names(contents: &str) -> BTreeSet<String> {
        let mut names = BTreeSet::new();
        let mut search = contents;
        let needle = "dictation.";
        while let Some(idx) = search.find(needle) {
            let after = &search[idx..];
            let token: String = after
                .chars()
                .take_while(|c| c.is_ascii_lowercase() || *c == '_' || *c == '.')
                .collect();
            search = &after[token.len().max(1)..];
            let Some(rest) = token.strip_prefix(needle) else {
                continue;
            };
            // Typed DaemonEvent names use underscores
            // (`dictation.completed` -> `dictation_completed`); normalize
            // the dotted doc form before comparison.
            if rest.is_empty() {
                continue;
            }
            if !rest.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
                continue;
            }
            names.insert(format!("dictation_{}", rest.replace('.', "_")));
        }
        names
    }

    #[test]
    fn collect_event_envelope_names_reads_daemon_event_match_arms() {
        let source = "
            Self::DictationSessionCreated { .. } => \"dictation_session_created\",
            Self::DictationCompleted { .. } => \"dictation_completed\",
            Self::EventsLost { .. } => \"events_lost\",
";
        let names = collect_event_envelope_names(source);
        assert_eq!(
            names,
            [
                "dictation_completed",
                "dictation_session_created",
                "events_lost"
            ]
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
            "every DaemonEvent::name() arm should be collected",
        );
    }

    #[test]
    fn collect_doc_dictation_event_names_filters_to_dictation_namespace_only() {
        let md = "\
- `dictation.session_created` — fired when the session is created.
- `dictation.completed` / `dictation.failed` — terminal pair.
- `compose.job_created` — out of scope for this guard.
- The `dictation` namespace is prose when it has no event suffix.
";
        let names = collect_doc_dictation_event_names(md);
        assert_eq!(
            names,
            [
                "dictation_completed",
                "dictation_failed",
                "dictation_session_created"
            ]
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
        );
    }

    /// Every dictation-namespace SSE event name a public-facing doc
    /// enumerates must correspond to a real `EventEnvelope::new(...)`
    /// emission in the daemon. Scans the architecture overview AND
    /// the project README, since both are operator-facing and either
    /// can drift independently; a contributor renaming an event
    /// might fix overview.html but forget the README example, or vice
    /// versa.
    ///
    /// Reverse direction (every emitted event is documented) is
    /// intentionally not enforced — the daemon emits several
    /// non-dictation events (`compose.job_created`,
    /// `transcription.completed`, `worker.providers_unavailable`)
    /// the docs do not promise to list.
    #[test]
    fn every_doc_dictation_event_resolves_to_an_event_envelope_emission() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let lib_source =
            std::fs::read_to_string(format!("{manifest}/../voicelayer-core/src/domain.rs"))
                .expect("read voicelayer-core domain.rs");
        let emitted = collect_event_envelope_names(&lib_source);
        assert!(
            !emitted.is_empty(),
            "expected at least one EventEnvelope::new emission in lib.rs",
        );

        let overview_doc = "../../docs/architecture/overview.html";
        let readme_doc = "../../README.md";
        let doc_paths: &[&str] = &[overview_doc, readme_doc];
        let mut total_doc_events = 0usize;
        let mut undocumented_in_code: Vec<String> = Vec::new();
        for rel in doc_paths {
            let abs = format!("{manifest}/{rel}");
            let contents =
                std::fs::read_to_string(&abs).unwrap_or_else(|err| panic!("read {abs}: {err}"));
            let doc_events = collect_doc_dictation_event_names(&contents);
            total_doc_events += doc_events.len();
            for event in doc_events.difference(&emitted) {
                // Typed DaemonEvent names drop the `dictation` namespace
                // prefix for pipeline-internal variants
                // (`dictation_probe_analyzed` -> `probe_analyzed`); the SSE
                // `event:` field carries the full variant name, so accept
                // both the exact form and the prefix-stripped form.
                let stripped = event.strip_prefix("dictation_").unwrap_or(event.as_str());
                if !emitted.contains(stripped) {
                    undocumented_in_code.push(format!("{rel}: `{event}`"));
                }
            }
        }

        assert!(
            total_doc_events > 0,
            "expected at least one dictation event reference across the scanned docs \
             (overview.html, README.md) — `collect_doc_dictation_event_names` may have \
             lost its anchor",
        );
        assert!(
            undocumented_in_code.is_empty(),
            "scanned docs mention dictation events the daemon does not emit:\n  - {}\n\n\
             Either rename the doc reference to match the actual event, drop the \
             mention, or wire up an `EventEnvelope::new(\"<name>\", ...)` emission \
             in crates/voicelayerd/src/.",
            undocumented_in_code.join("\n  - "),
        );
    }
}
