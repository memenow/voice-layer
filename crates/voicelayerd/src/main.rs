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
    run_daemon(DaemonConfig::new(args.socket_path, args.project_root)).await?;
    Ok(())
}
