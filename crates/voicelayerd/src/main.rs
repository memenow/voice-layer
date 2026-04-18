use std::path::PathBuf;

use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use voicelayerd::{DaemonConfig, default_project_root, default_socket_path, run_daemon};

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
    let socket_path = args.socket_path.unwrap_or_else(default_socket_path);
    let project_root = args.project_root.unwrap_or_else(default_project_root);

    run_daemon(DaemonConfig::with_project_root(socket_path, project_root)).await?;
    Ok(())
}
