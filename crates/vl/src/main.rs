mod cli;
mod config;
mod foreground_ptt;
#[cfg(target_os = "macos")]
mod injection;
mod terminal_targets;

#[tokio::main]
async fn main() {
    if let Err(error) = cli::run().await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
