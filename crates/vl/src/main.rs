mod cli;
mod config;
mod foreground_ptt;
mod preview;
mod terminal_targets;
mod uds;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    cli::run().await
}
