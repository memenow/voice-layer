//! Thin entry point for the `voicelayerd` binary; the daemon's real
//! lifecycle lives in [`voicelayerd::run_daemon`] inside the library
//! crate so integration tests can drive it without a process boundary.

use std::path::PathBuf;

use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use voicelayerd::{DaemonConfig, run_daemon};

#[derive(Debug, Parser)]
#[command(name = "voicelayerd")]
#[command(about = "VoiceLayer local daemon")]
struct Args {
    #[arg(long, env = "VOICELAYER_SOCKET_PATH")]
    socket_path: Option<PathBuf>,
    #[arg(long, env = "VOICELAYER_PROJECT_ROOT")]
    project_root: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();
    // Explicit CLI args win over the config file; both fall back to the
    // platform defaults resolved inside DaemonConfig.
    let settings = voicelayer_core::VoiceLayerConfig::load().unwrap_or_default();
    run_daemon(DaemonConfig::with_settings(
        args.socket_path,
        args.project_root,
        settings,
    ))
    .await?;
    Ok(())
}
