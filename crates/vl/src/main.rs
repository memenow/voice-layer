//! Thin entry point for the `vl` operator CLI; argument parsing and the
//! actual command dispatch live in [`cli::run`] so library-style tests
//! can drive each subcommand without forking a child process.

mod cli;
mod config;
mod foreground_ptt;
mod terminal_targets;
mod tui_glass;
mod uds;

#[tokio::main]
async fn main() {
    if let Err(error) = cli::run().await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
