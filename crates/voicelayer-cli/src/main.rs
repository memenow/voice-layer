//! Entry point for the `vl` CLI and terminal push-to-talk panel.

mod commands;
mod config;
mod panel_style;
mod ptt;
mod terminal_targets;

#[tokio::main]
async fn main() {
    if let Err(error) = commands::run().await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
